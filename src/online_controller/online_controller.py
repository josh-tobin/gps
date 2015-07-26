import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp
import scipy.linalg
import logging
import time

from algorithm.policy.policy import Policy
from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy

LOGGER = logging.getLogger(__name__)

class OnlineController(Policy):
    def __init__(self, dX, dU, dynprior, cost, offline_fd=None, offline_fc=None, offline_dynsig=None):
        self.dynprior = dynprior
        self.LQR_iter = 1
        self.dX = dX
        self.dU = dU
        self.cost = cost
        self.gamma = 0.5
        self.maxT = 100
        self.min_mu = 1e-4 
        self.del0 = 2
        self.NSample = 1

        self.H = 10
        self.empsig_N = 3
        self.sigreg = 1e-6

        self.prevX = None
        self.prevU = None
        self.prev_policy = None

        self.offline_fd = offline_fd
        self.offline_fc = offline_fc
        self.offline_dynsig = offline_dynsig

    def act_pol(self, x, empmu, empsig, prevx, prevu, t):
        #start = time.time()
        dX = self.dX
        dU = self.dU
        #gamma = self.gamma
        self.mu = empmu
        self.sigma = empsig
        self.prevX = prevx
        self.prevU = prevu

        if t==0:
            # Execute something for first action.
            H = self.H
            K = np.zeros((H, dU, dX))
            k = np.zeros((H, dU))
            cholPSig = np.zeros((H, dU, dU))
            self.prev_policy = LinearGaussianPolicy(K, k, None, cholPSig, None)
            return self.prev_policy

        #pt = np.r_[self.prevX,self.prevU,x]
        #self.mu = self.mu*self.gamma + pt*(1-self.gamma)
        #self.sigma = self.sigma*self.gamma + np.outer(pt,pt)*(1-self.gamma)
        #empsig = self.sigma

        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            lgpolicy, reg_mu, reg_del = self.lqr(t, x, self.prev_policy, empsig, reg_mu, reg_del)

            # Store traj
            self.prev_policy = lgpolicy
        #self.prev_policy.K[1] = self.prev_policy.K[0]
        #self.prev_policy.k[1] = self.prev_policy.k[0]
        u = self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]
        print u
        # Store state and action.
        return self.prev_policy
 

    def act(self, x, obs, t, noise):
        #start = time.time()
        dX = self.dX
        dU = self.dU
        gamma = self.gamma

        if t==0:
            # Execute something for first action.
            H = self.H
            K = np.zeros((H, dU, dX))
            k = np.zeros((H, dU))
            cholPSig = np.zeros((H, dU, dU))
            U = K[0].dot(x) + k[0] #+ cholPSig[t].dot(np.random.randn(dU));

            self.prevU = U;
            self.prevX = x;

            self.prev_policy = LinearGaussianPolicy(K, k, None, cholPSig, None)

            pt = np.r_[self.prevX,self.prevU,x]
            self.mu = pt
            self.sigma = np.outer(pt,pt)
            return U

        # Update mean and covariance.
        # Since this works well *without* subtracting mean, could the same trick
        # work well with mfcgps GMM prior?
        pt = np.r_[self.prevX,self.prevU,x]
        self.mu = self.mu*gamma + pt*(1-gamma)
        self.sigma = self.sigma*gamma + np.outer(pt,pt)*(1-gamma)
        empsig = self.sigma

        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            lgpolicy, reg_mu, reg_del = self.lqr(t, x, self.prev_policy, empsig, reg_mu, reg_del)

            # Store traj
            self.prev_policy = lgpolicy

        #TODO: Re-enable noise once this works.
        u = self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]

        # Store state and action.
        self.prevX = x
        self.prevU = u
        #elapsed = time.time()-start
        #print 'Controller Act:', elapsed
        return u

    def lqr(self, T, x, lgpolicy, empsig, reg_mu, reg_del):
        dX = self.dX
        dU = self.dU
        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        horizon = min(self.H, self.maxT-T);

        cv, Cm, Fd, fc = self.estimate_cost(horizon, x, lgpolicy, empsig, T)

        # Compute optimal action with short horizon MPC.
        fail = True;
        decrease_mu = True;
        del0 = self.del0;
        min_mu = self.min_mu;
        K = np.zeros((horizon,dU, dX))
        cholPSig = np.zeros((horizon, dU, dU))
        k = np.zeros((horizon, dU))
        while fail:
            Vxx = np.zeros((dX, dX))
            Vx = np.zeros(dX)
            fail = False
            for t in range(horizon-1, -1, -1):
                F = Fd[t]
                f = fc[t]

                Qtt = Cm[t]
                Qt = cv[t]
                
                Vxx = Vxx + reg_mu*np.eye(dX)
                Qtt = Qtt + F.T.dot(Vxx).dot(F)
                Qt = Qt + F.T.dot(Vx+Vxx.dot(f))

                Qtt = 0.5*(Qtt+Qtt.T)

                try:
                    U = sp.linalg.cholesky(Qtt[iu,iu], check_finite=False)
                    L = U.T
                except LinAlgError:
                    fail = True
                    decrease_mu = False
                    break

                K[t] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, Qtt[iu, ix], lower=True, check_finite=False), check_finite=False)
                k[t] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, Qt[iu], lower=True, check_finite=False), check_finite=False)
                pol_covar = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True, check_finite=False), check_finite=False)
                cholPSig[t] = sp.linalg.cholesky(pol_covar, check_finite=False)

                # Compute value function.
                Vxx = Qtt[ix, ix] + Qtt[ix, iu].dot(K[t])
                Vx = Qt[ix] + Qtt[ix, iu].dot(k[t])
                Vxx = 0.5 * (Vxx + Vxx.T)

            #Tassa regularization scheme
            if fail:
                if reg_mu > 1e5:
                    raise ValueError("Failed to find SPD solution")
                else:
                    reg_del = max(del0, reg_del*del0)
                    reg_mu = max(min_mu, reg_mu*reg_del)
                    LOGGER.debug('Increasing mu -> %f', reg_mu)
            elif decrease_mu:
                reg_del = min(1/del0, reg_del/del0)
                delmu = reg_del*reg_mu
                if delmu > min_mu:
                    reg_mu = delmu;
                else:
                    reg_mu = min_mu;
                LOGGER.debug('Decreasing mu -> %f', reg_mu)

        policy = LinearGaussianPolicy(K, k, None, cholPSig, None)
        return policy, reg_mu, reg_del        

    def estimate_cost(self, horizon, x0, lgpolicy, empsig, cur_timestep):
        """
        Returns cost matrices and dynamics
        """
        # Cost + dynamics estimation

        H = horizon

        N = self.NSample;

        cholPSig = lgpolicy.chol_pol_covar
        #PSig = lgpolicy.pol_covar
        K = lgpolicy.K
        k = lgpolicy.k

        dX = K.shape[2]
        dU = K.shape[1]
        dT = dX+dU
        ix = slice(dX)
        iu = slice(dX, dX+dU)

        #Run forward pass

        # Allocate space.
        trajsig = np.zeros((H,dT, dT))
        mu = np.zeros((H,dT))

        trajsig[0,ix,ix] = np.zeros((dX, dX))
        mu[0,ix] = x0;

        F = np.zeros((H, dX, dT))
        f = np.zeros((H, dX))
        dynsig = np.zeros((H,dX, dX))
        F[0], f[0], dynsig[0] = self.getdynamics(self.prevX, self.prevU, x0, empsig, cur_timestep);

        K = lgpolicy.K
        k = lgpolicy.k
        # Perform forward pass.
        for t in range(H):
            PSig = cholPSig[t].T.dot(cholPSig[t])
            trajsig[t] = np.r_[
                                np.c_[trajsig[t,ix,ix], trajsig[t,ix,ix].dot(K[t].T)],
                                np.c_[K[t].dot(trajsig[t,ix,ix]), K[t].dot(trajsig[t,ix,ix]).dot(K[t].T) + PSig]
                             ]
            mu[t] = np.r_[mu[t,ix], K[t].dot(mu[t,ix]) + k[t]]

            # Reuse old dynamics
            F[t] = F[0]
            f[t] = f[0]
            dynsig[t] = dynsig[0]

            if t < H-1:
                # Estimate new dynamics here based on mu
                #F[t], f[t], dynsig[t] = self.getdynamics(mu[t,ix], mu[t,iu], mu[t+1, ix], empsig, cur_timestep+t);
                trajsig[t+1,ix,ix] = F[t].dot(trajsig[t]).dot(F[t].T) + dynsig[t]
                mu[t+1,ix] = F[t].dot(mu[t]) + f[t]

        #cc = zeros(1,horizon,N);
        cv = np.zeros((N, H, dT))
        Cm = np.zeros((N, H, dT, dT))
        #Xs, Us = self.trajsamples(dX, dU, H, mu, trajsig, lgpolicy, N)

        for n in range(N):
            # Get costs.
            #l,lx,lu,lxx,luu,lux = self.cost.eval(Xs[n],Us[n], cur_timestep);
            l,lx,lu,lxx,luu,lux = self.cost.eval(mu[:,ix],mu[:,iu], cur_timestep);
            #[cc(:,:,i),lx,lu,lxx,luu,lux] = controller.cost.eval(Xs(:,:,i),Us(:,:,i),[],cost_infos(:,:,i));

            #cs(:,:,i) = cc(:,:,i);
            # Assemble matrix and vector.
            cv[n] = np.c_[lx,lu]
            Cm[n] = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

            # Adjust for expanding cost around a sample.
            #yhat = np.c_[Xs[n], Us[n]]
            yhat = np.c_[mu[:,ix], mu[:,iu]]
            rdiff = -yhat  # T x (X+U)
            rdiff_expand = np.expand_dims(rdiff, axis=2)  # T x (X+U) x 1
            cv_update = np.sum(Cm[n] * rdiff_expand, axis=1)  # T x (X+U)
            #cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n] += cv_update      

        #cc = mean(cc,3);
        cv = np.mean(cv, axis=0)
        Cm = np.mean(Cm, axis=0)
        return cv, Cm, F, f

    def trajsamples(self, dX, dU, T, mu, sigma, lgpolicy, N):
        """
        Compute samples
        """
        # Constants.
        ix = slice(dX)
        iu = slice(dX, dX+dU)

        # Allocate space.
        pX = np.zeros((N,T, dX))
        pU = np.zeros((N,T, dU))
        #pProb = np.zeros(1,T,N);

        for t in range(T):
            samps = np.random.randn(dX,N);
            sigma[t, ix,ix] = 0.5*(sigma[t,ix,ix]+sigma[t,ix,ix].T)

            # Fix small eigenvalues only.
            #[val, vec] = np.linalg.eig(sigma[t,ix,ix])
            #val0 = val;
            #val = np.real(val)
            #val = np.maximum(val,1e-6);
            #sigma[t, ix,ix] = vec.dot(np.diag(val)).dot(vec.T)
            sigma[t, ix, ix] += 1e-6*np.eye(dX)  # Much faster

            # Store sample probabilities.
            #pProb[:,t,:] = -0.5*np.sum(samps**2,axis=0) - 0.5*np.sum(np.log(val));

            # Draw samples.
            samps = sp.linalg.cholesky(sigma[t,ix,ix], overwrite_a=True, check_finite=False).T.dot(samps)+np.expand_dims(mu[t, ix], axis=1)

            pX[:,t,:] = samps.T #np.expand_dims(samps, axis=1)
            pU[:,t,:] = (lgpolicy.K[t].dot(samps)+np.expand_dims(lgpolicy.k[t], axis=1) + \
                lgpolicy.chol_pol_covar[t].dot(np.random.randn(dU, N))).T
        return pX, pU

    def getdynamics(self, prev_x, prev_u, cur_x, empsig, t):
        """
        """
        dX = self.dX
        dU = self.dU

        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)

        nearest_idx = self.cost.compute_nearest_neighbors(prev_x.reshape(1,dX), prev_u.reshape(1,dU), t)[0]

        offline_fd = self.offline_fd[nearest_idx]
        offline_fc = self.offline_fc[nearest_idx]
        offline_dynsig = self.offline_dynsig[nearest_idx]
        #return offline_fd, offline_fc, offline_dynsig

        xux = np.r_[prev_x, prev_u, cur_x]
        #xu = np.r_[prev_x, prev_u]
        mu0,Phi,m,n0 = self.dynprior.eval(dX, dU, xux.reshape(1, dX+dU+dX))

        N = self.empsig_N
        mun = self.mu

        sigma = (N*empsig + Phi + ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
        
        #sigma = Phi/m;  # Prior only
        #sigma = empsig;  % Moving average only
        #controller.sigma = sigma;  % TODO: Update controller.sigma here?
        sigma[it, it] = sigma[it, it] + self.sigreg*np.eye(dX+dU)

        Fm = (np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip])).T
        fv = mun[ip] - Fm.dot(mun[it]);

        dyn_covar = sigma[ip,ip] - Fm.dot(sigma[it,it]).dot(Fm.T)
        dyn_covar = 0.5*(dyn_covar+dyn_covar.T)  # Make symmetric

        import pdb; pdb.set_trace()
        Fm = 0.5*Fm+0.5*offline_fd
        fv = 0.5*fv+0.5*offline_fc
        dyn_covar = 0.5*dyn_covar + 0.5*offline_dynsig
        return Fm, fv, dyn_covar
