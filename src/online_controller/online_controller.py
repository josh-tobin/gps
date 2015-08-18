import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp
import scipy.linalg
import logging
import time

from algorithm.policy.policy import Policy
from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy

import train_dyn_net as theano_dynamics

LOGGER = logging.getLogger(__name__)

class OnlineController(Policy):
    def __init__(self, dX, dU, dynprior, cost, maxT = 100, ee_sites=None, dyn_init_mu=None, dyn_init_sig=None, offline_K=None, offline_k=None, offline_fd=None, offline_fc=None, offline_dynsig=None):
        self.dynprior = dynprior
        self.LQR_iter = 1
        self.dX = dX
        self.dU = dU
        self.cost = cost
        self.gamma = 0.05
        self.maxT = maxT
        self.min_mu = 1e-6 
        self.del0 = 2
        self.NSample = 1

        self.H = 10
        self.empsig_N = 3
        self.sigreg = 1e-6
        self.time_varying_dynamics = False

        self.prevX = None
        self.prevU = None
        self.prev_policy = None
        self.u_noise = 5.0

        self.offline_fd = offline_fd
        self.offline_fc = offline_fc
        self.offline_dynsig = offline_dynsig
        self.ee_sites = ee_sites
        self.dyn_init_mu = dyn_init_mu
        self.dyn_init_sig = dyn_init_sig
        self.use_prior_dyn = False

        self.nn_dynamics = True
        self.copy_offline_traj = False
        self.offline_K = offline_K
        self.offline_k = offline_k
        self.inputs = []
        self.calculated = []
        self.fwd_hist = [None]*self.maxT

        #self.dyn_net = theano_dynamics.get_net('trap_contact_full_state.pkl') #theano_dynamics.load_net('norm_net.pkl')
        #self.dyn_net_ls = theano_dynamics.get_net('net/trap_contact_small.pkl') #theano_dynamics.load_net('norm_net.pkl')

        #self.dyn_net = theano_dynamics.get_net('net/plane_relu3.pkl')
        netname = 'net/plane_relu5.pkl'
        self.dyn_net = theano_dynamics.get_net(netname)
        self.dyn_net_ref = theano_dynamics.get_net(netname)

        #self.dyn_net = theano_dynamics.get_net('net/mjc_lsq_air.pkl')
        #self.dyn_net = theano_dynamics.get_net('net/mjcnet_lsq.pkl')
        #self.dyn_net = theano_dynamics.get_net('net/mjcnet_lsq.pkl')
        #self.dyn_net_ref = theano_dynamics.get_net('net/mjcnet_relu.pkl')
        #self.dyn_net = theano_dynamics.get_net('trap_contact_small.pkl')

        self.vis_forward_pass_joints = None  # Holds joint state for visualizing forward pass
        self.vis_forward_ee = None

    def act_pol(self, x, empmu, empsig, prevx, prevu, sitejac, eejac, t):
        if t == 0:
            self.inputs = [] #debugging

        dX = self.dX 
        dU = self.dU
        #gamma = self.gamma
        self.prevX = prevx
        self.prevU = prevu
        self.inputs.append({'x':x, 'empmu':empmu, 'empsig':empsig, 'prevx':prevx, 'prevu':prevu, 'eejac':eejac, 't':t})

        if t==0:
            # Execute something for first action.
            H = self.H
            K = np.zeros((H, dU, dX))
            k = np.zeros((H, dU))
            cholPSig = np.zeros((H, dU, dU))

            self.mu = self.dyn_init_mu
            self.sigma = self.dyn_init_sig
            self.prev_policy = LinearGaussianPolicy(K, k, None, cholPSig, None)
            if self.copy_offline_traj:
                for i in range(t,t+self.prev_policy.T):
                    self.prev_policy.K[i-t] = self.offline_K[i]
                    self.prev_policy.k[i-t] = self.offline_k[i]
            return self.prev_policy

        if self.nn_dynamics:
            self.update_nn_dynamics(prevx, prevu, x)
        else:
            self.update_emp_dynamics(prevx, prevu, x)
            self.prior_dynamics_heuristic_switch(self.prevX, self.prevU, x)

        #pt = np.r_[self.prevX,self.prevU,x]
        #self.mu = self.mu*self.gamma + pt*(1-self.gamma)
        #self.sigma = self.sigma*self.gamma + np.outer(pt,pt)*(1-self.gamma)
        #empsig = self.sigma

        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            lgpolicy, reg_mu, reg_del = self.lqr(t, x, self.prev_policy, reg_mu, reg_del)

            # Store traj
            self.prev_policy = lgpolicy
        
        #Trick
        """
        self.prev_policy.T = 2;
        new_K = np.zeros((2, dU, dX))
        new_k = np.zeros((2, dU))
        new_K[0] = self.prev_policy.K[0]
        new_K[1] = self.prev_policy.K[0]
        new_k[0] = self.prev_policy.k[0]
        new_k[1] = self.prev_policy.k[0]
        self.prev_policy.K = new_K
        self.prev_policy.k = new_k
        """
        
        u = self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]
        self.calculated.append({
            'u':u, 'K':np.copy(self.prev_policy.K), 'k':np.copy(self.prev_policy.k), 't':t
            })

        #print 'Online:', u
        #u = self.offline_K[t].dot(x)+self.offline_k[t]
        #print 'Offline:', u
        if self.copy_offline_traj:
            print 'CopyTraj!'
            newK = np.copy(self.prev_policy.K)
            newk = np.copy(self.prev_policy.k)
            for i in range(t,t+self.prev_policy.T):
                newK[i-t] = self.offline_K[i]
                newk[i-t] = self.offline_k[i]
            return LinearGaussianPolicy(newK, newk, None, None, None)
        #self.prev_policy.K.fill(0.0)
        #self.prev_policy.k.fill(0.0)

        # Generate noise - jacobian transpose method
        noise_dir = np.array([0,0,1])
        noise_vec = eejac[0:3,:].T.dot(noise_dir)
        noise_vec = noise_vec/np.linalg.norm(noise_vec)
        noise_covar = np.outer(noise_vec,noise_vec)
        final_noise = noise_covar.dot(np.random.randn(7))
        noise_vec = self.u_noise*final_noise

        # Generate noise - 
        #noise_vec = self.u_noise*np.random.randn(7)

        self.prev_policy.k[0] += noise_vec
        if self.prev_policy.T > 1:
            self.prev_policy.k[1] += noise_vec

        # Store state and action.
        return self.prev_policy 

    def act(self, x, obs, t, noise):
        print 'T=', t
        #start = time.time()
        dX = self.dX
        dU = self.dU
        gamma = self.gamma

        if t==0:
            # Execute something for first action.
            H = self.H
            K = self.offline_K[0:H,:] #np.zeros((H, dU, dX))
            k = self.offline_k[0:H,:] #np.zeros((H, dU))
            k += np.random.randn(H, dU)*0.01
            cholPSig = np.zeros((H, dU, dU))
            U = K[0].dot(x) + k[0] #+ cholPSig[t].dot(np.random.randn(dU));
            U.fill(0.0)

            self.prevU = U;
            self.prevX = x;

            self.prev_policy = LinearGaussianPolicy(K, k, None, cholPSig, None)

            #pt = np.r_[self.prevX,self.prevU,x]
            #self.mu = pt
            #self.sigma = np.outer(pt,pt)
            self.mu = self.dyn_init_mu
            self.sigma = self.dyn_init_sig
            return U

        # Update mean and covariance.
        # Since this works well *without* subtracting mean, could the same trick
        # work well with mfcgps GMM prior?
        if self.nn_dynamics:
            self.update_nn_dynamics(self.prevX, self.prevU, x)
        else:
            self.update_emp_dynamics(self.prevX, self.prevU, x)
            self.prior_dynamics_heuristic_switch(self, self.prevX, self.prevU, x)

        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            lgpolicy, reg_mu, reg_del = self.lqr(t, x, self.prev_policy, reg_mu, reg_del)

            # Store traj
            self.prev_policy = lgpolicy

        #TODO: Re-enable noise once this works.
        if self.copy_offline_traj:
            u = self.offline_K[t].dot(x)+self.offline_k[t]
        else:
            u = self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]
            u += self.prev_policy.chol_pol_covar[0].dot(self.u_noise*np.random.randn(7))

        print u
        # Store state and action.
        self.prevX = x
        self.prevU = u
        #elapsed = time.time()-start
        #print 'Controller Act:', elapsed
        return u

    def prior_dynamics_heuristic_switch(self, prevx, prevu, curx):
        dX = self.dX
        dU = self.dU
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        pt = np.r_[prevx,prevu]

        empFm = (np.linalg.pinv(self.sigma[it, it]).dot(self.sigma[it, ip])).T
        empfv = self.mu[ip] - empFm.dot(self.mu[it]);
        emperr = np.linalg.norm(empFm.dot(pt) + empfv - curx)

        priorFm = (np.linalg.pinv(self.dyn_init_sig[it, it]).dot(self.dyn_init_sig[it, ip])).T
        priorfv = self.dyn_init_mu[ip] - priorFm.dot(self.dyn_init_mu[it]);
        priorerr = np.linalg.norm(priorFm.dot(pt) + priorfv - curx)
        print 'PRIOR?:', self.use_prior_dyn,' emperr:', emperr, ' // priorerr:', priorerr

        if self.use_prior_dyn:
            if emperr < priorerr:
                print 'Switching to emp dynamics!'
                self.use_prior_dyn = False
        else:
            if emperr > 2*priorerr:
                print 'Switching to prior dynamics!'
                self.use_prior_dyn = True

    def update_emp_dynamics(self, prevx, prevu, cur_x):
        pt = np.r_[prevx,prevu,cur_x]
        gamma = self.gamma
        self.mu = self.mu*(1-gamma) + pt*(gamma)
        pt = pt-self.mu
        self.sigma = self.sigma*(1-gamma) + np.outer(pt,pt)*(gamma)
        self.sigma = 0.5*(self.sigma+self.sigma.T)

    def update_nn_dynamics(self, prevx, prevu, cur_x):
        pt = np.r_[prevx, prevu]
        lbl = cur_x
        for i in range(0):
            # Lsq use 0.003
            print 'NN Dynamics Loss: %f // Ref:%f' % ( self.dyn_net.train_single(pt, lbl, lr=0.001, momentum=0.99), self.dyn_net_ref.obj_vec(pt, lbl))

    def lqr(self, T, x, lgpolicy, reg_mu, reg_del):
        dX = self.dX
        dU = self.dU
        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        horizon = min(self.H, self.maxT-T);

        cv, Cm, Fd, fc = self.estimate_cost(horizon, x, lgpolicy, T)

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
                #Qtt = Qtt + F.T.dot(Vxx.dot(F))
                Qtt = Qtt + F.T.dot(Vxx.dot(F))
                Qt = Qt + F.T.dot(Vx)+F.T.dot(Vxx).dot(f)

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

    def estimate_cost(self, horizon, x0, lgpolicy, cur_timestep):
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
        #trajsig = np.zeros((H,dT, dT))
        mu = np.zeros((H,dT))

        #trajsig[0,ix,ix] = np.zeros((dX, dX))
        mu[0,ix] = x0;

        F = np.zeros((H, dX, dT))
        f = np.zeros((H, dX))
        dynsig = np.zeros((H,dX, dX))

        K = lgpolicy.K
        k = lgpolicy.k
        # Perform forward pass.
        for t in range(H):
            PSig = cholPSig[t].T.dot(cholPSig[t])
            #trajsig[t] = np.r_[
            #                    np.c_[trajsig[t,ix,ix], trajsig[t,ix,ix].dot(K[t].T)],
            #                    np.c_[K[t].dot(trajsig[t,ix,ix]), K[t].dot(trajsig[t,ix,ix]).dot(K[t].T) + PSig]
            #                 ]
            cur_action = K[t].dot(mu[t,ix])+k[t]
            mu[t] = np.r_[mu[t,ix], cur_action]
            #mu[t] = np.r_[mu[t,ix], np.zeros(dU)]

            # Reuse old dynamics
            if not self.time_varying_dynamics:
                if t==0: 
                    F[0], f[0], dynsig[0] = self.getdynamics(self.prevX, self.prevU, x0, cur_timestep, cur_action=cur_action);
                F[t] = F[0]
                f[t] = f[0]
                dynsig[t] = dynsig[0]

            if t < H-1:
                # Estimate new dynamics here based on mu
                if self.time_varying_dynamics:
                    F[t], f[t], dynsig[t] = self.getdynamics(mu[t-1,ix], mu[t-1,iu], mu[t, ix], cur_timestep+t, cur_action=cur_action);
                #trajsig[t+1,ix,ix] = F[t].dot(trajsig[t]).dot(F[t].T) + dynsig[t]
                mu[t+1,ix] = F[t].dot(mu[t]) + f[t]
        self.fwd_hist[cur_timestep] = {'trajmu': mu, 'F': F, 'f': f}
        self.vis_forward_pass_joints = mu[:,0:7]
        self.vis_forward_ee = mu[:,21:30]

        #cc = np.zeros((N, H))
        cv = np.zeros((N, H, dT))
        Cm = np.zeros((N, H, dT, dT))
        #Xs, Us = self.trajsamples(dX, dU, H, mu, trajsig, lgpolicy, N)

        for n in range(N):
            # Get costs.
            #l,lx,lu,lxx,luu,lux = self.cost.eval(Xs[n],Us[n], cur_timestep);
            #newmu = np.zeros_like(mu[:,iu]) 
            newmu = mu[:,iu]
            l,lx,lu,lxx,luu,lux = self.cost.eval(mu[:,ix], newmu, cur_timestep);
            #[cc(:,:,i),lx,lu,lxx,luu,lux] = controller.cost.eval(Xs(:,:,i),Us(:,:,i),[],cost_infos(:,:,i));

            #cs(:,:,i) = cc(:,:,i);
            # Assemble matrix and vector.

            #cc[n] = l
            self.prev_cv = np.c_[lx, lu] #TEMP
            self.prev_cc = l #TEMP
            cv[n] = np.c_[lx, lu]
            Cm[n] = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

            # Adjust for expanding cost around a sample.
            yhat = mu #np.c_[mu[:,ix], newmu_u]
            rdiff = -yhat  # T x (X+U)
            #if self.ref is not None:
            #    rdiff = self.ref[cur_timestep:cur_timestep+H]-yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)  # T x (X+U) x 1
            cv_update = np.sum(Cm[n] * rdiff_expand, axis=1)  # T x (X+U)
            #cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            #self.cc = cc  # TEMP
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

    def getdynamics(self, prev_x, prev_u, cur_x, t, cur_action=None):
        """
        """
        dX = self.dX
        dU = self.dU

        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)

        xu = np.r_[cur_x, cur_action].astype(np.float32)
        if self.nn_dynamics:
            #nearest_idx = self.cost.compute_nearest_neighbors(prev_x.reshape(1,dX), prev_u.reshape(1,dU), t)[0]
            #offline_fd = self.offline_fd[t]
            #offline_fc = self.offline_fc[t]
            #offline_dynsig = self.offline_dynsig[t]
            #return offline_fd, offline_fc, offline_dynsig

            dynsig = np.eye(dX)
            F, f = self.dyn_net.getF(xu)

            return F, f, dynsig
        else:
            xux = np.r_[prev_x, prev_u, cur_x]
            #xu = np.r_[prev_x, prev_u]

            N = self.empsig_N

            if self.use_prior_dyn:
                mun = self.dyn_init_mu
                empsig = self.dyn_init_sig
            else:
                mun = self.mu
                empsig = self.sigma

            mu0,Phi,m,n0 = self.dynprior.eval(dX, dU, xux.reshape(1, dX+dU+dX))
            sigma = (N*empsig + Phi + ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
            #sigma = empsig;  # Moving average only
            #controller.sigma = sigma;  % TODO: Update controller.sigma here?
            sigma[it, it] = sigma[it, it] + self.sigreg*np.eye(dX+dU)

            #(np.linalg.pinv(empsig[it, it]).dot(empsig[it, ip])).T
            Fm = (np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip])).T
            fv = mun[ip] - Fm.dot(mun[it]);

            #Fm[7:14] = F2[7:14]
            #fv[7:14] = f2[7:14]

            #Fms = (np.linalg.pinv(self.sigma[it, it]).dot(self.sigma[it, ip])).T

            dyn_covar = sigma[ip,ip] - Fm.dot(sigma[it,it]).dot(Fm.T)
            dyn_covar = 0.5*(dyn_covar+dyn_covar.T)  # Make symmetric

            return Fm, fv, dyn_covar

    def get_forward_joint_states(self):
        """ Joint states for visualizing forward pass in RVIZ
            Returns an H x 7 array
        """
        return self.vis_forward_pass_joints

    def get_forward_end_effector(self, pnt=0):
        """ Joint states for visualizing forward pass in RVIZ
            Returns an H x 3 array
        """
        if self.vis_forward_ee is not None:
            return self.vis_forward_ee[:,pnt*3:pnt*3+3]