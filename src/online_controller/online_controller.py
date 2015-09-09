import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp
import scipy.linalg
import logging
import time
import nnvis
import theano_rnn
from proto.gps_pb2 import *
from agent.agent_utils import generate_noise

from algorithm.policy.policy import Policy
from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy

import train_dyn_net as theano_dynamics

LOGGER = logging.getLogger(__name__)

class OnlineController(Policy):
    def __init__(self, dX, dU, dynprior, cost, gp=None, maxT = 100, use_ee_sites=True, ee_idx=None, ee_sites=None, jac_service=None, dyn_init_mu=None, dyn_init_sig=None, offline_K=None, offline_k=None, offline_fd=None, offline_fc=None, offline_dynsig=None):
        self.dynprior = dynprior
        self.offline_fd = offline_fd
        self.offline_fc = offline_fc
        self.offline_dynsig = offline_dynsig
        self.ee_sites = ee_sites
        self.ee_idx = ee_idx
        self.use_ee_sites = use_ee_sites
        self.jac_service = jac_service
        self.gp = gp
        self.dyn_init_mu = dyn_init_mu
        self.dyn_init_sig = dyn_init_sig/100 + 1e-5*np.eye(dX+dU+dX)
        #dyn_init_logdet = 2*sum(np.log(np.diag(sp.linalg.cholesky(self.dyn_init_sig+ 1e-6*np.eye(self.dyn_init_sig.shape[0])))))

        # Rescale initial covariance
        #self.dyn_init_sig = self.dyn_init_sig / (1.0 * np.exp(dyn_init_logdet))

        #self.dyn_init_mu.fill(0.0)
        #self.dyn_init_sig.fill(0.0)
        self.dX = dX
        self.dU = dU
        self.cost = cost
        self.maxT = maxT
        self.prevX = None
        self.prevU = None
        self.prev_policy = None
        self.noise = None
        self.rnn_hidden_state = None
        self.adaptive_gamma_logprob = None
        self.restricted_context_list = []
        self.offline_K = offline_K
        self.offline_k = offline_k

        # Algorithm Settings
        self.H = 17 # Horizon

        # LQR
        self.LQR_iter = 1  # Number of LQR iterations to take
        self.min_mu = 1e-6 # LQR regularization
        self.del0 = 2 # LQR regularization

        # KL Div constraint
        self.eta = 0.01 # Initial eta for DGD w/ KL-div constrant
        self.use_kl_constraint = False

        # Noise scaling
        self.u_noise = 0.03 # Noise to add

        #Dynamics settings
        self.adaptive_gamma = False
        self.gamma = 0.05  # Moving average parameter
        self.empsig_N = 3.0 # Weight of least squares vs GMM/NN prior
        self.sigreg = 1e-5 # Regularization on dynamics covariance
        self.time_varying_dynamics = False
        self.use_prior_dyn = False
        self.gmm_prior = False
        self.lsq_prior = True
        self.gp_prior = False
        self.mix_prior_strength = 1.0

        #Neural net options
        self.nn_dynamics = False  # If TRUE, uses neural network for dynamics. Else, uses moving average least squares
        self.nn_prior = False # If TRUE and nn_dynamics is on, mixes moving average least squares with neural network as a prior
        self.nn_update_iters = 0  # Number of SGD iterations to take per timestep
        self.nn_lr = 0.0004  # SGD learning rate
        self.nn_recurrent = False  # Set to true if network is recurrent. Turns on RNN hidden state management
        self.nn_recurrent_plot = False  # Turn on plotting for recurrent hidden state
        self.restricted_context = 0  # 0 means off. Otherwise keeps a history of state, action pairs

        #Other
        self.copy_offline_traj = False  # If TRUE, overrides calculated controller with offline controller. Useful for debugging

        self.inputs = []
        self.calculated = []
        self.fwd_hist = [{} for _ in range(self.maxT)]

        if self.nn_dynamics:
            self.update_hidden_state = True
            #netname = 'net/combined_100.pkl'
            #netname = 'net/combined_50_30_nowkbench.pkl'
            #netname = 'net/combined_50_30_halfworkbench.pkl'
            netname = 'net/combined_100_nowkbench.pkl'
            #netname = 'net/rnn/robo_net15_nogear.pkl.ff'

            #netname = 'net/mjc_simplegate.pkl.ff'  # Works well on pos 9
            #netname = 'net/rnn/net7.pkl.ff'   # Works OKAY
            #netname = 'net/rnn/net10.pkl.ff'
            #netname = 'net/rnn/net15.pkl.ff'

            if self.nn_recurrent:
                self.dyn_net = theano_rnn.unpickle_net(netname)
                self.dyn_net.update(stage='test')
                self.dyn_net.init_functions(output_blob='acc')
                self.rnn_hidden_state = self.dyn_net.get_init_hidden_state()

                if self.nn_recurrent_plot:
                    #gph1 = nnvis.ImagePlot(80, name='hidden1')
                    #gph2 = nnvis.ImagePlot(40, name='hidden1')
                    gphs = []
                    for state in self.rnn_hidden_state:
                        gphs.append(nnvis.GraphPlot(state.shape[0], -5, 20, name='hidden2'))
                    self.nn_vis = nnvis.NNVis(gphs, 100)
            else:
                self.dyn_net = theano_dynamics.get_net(netname, rec=True, dX=self.dX, dU=self.dU)
            #if self.nn_update_iters>0:  # Keep a reference for overfitting
            #    self.dyn_net_ref = theano_dynamics.get_net(netname, rec=rec, dX=32+6, dU=7)

        #RVIZ stuff
        self.vis_forward_pass_joints = None  # Holds joint state for visualizing forward pass
        self.vis_forward_ee = None
        self.rviz_traj = [None]*self.maxT

    def act_pol(self, x, empmu, empsig, prevx, prevu, sitejac, eejac, t):
        """
        Return a policy to execute rather than an action.
        Used by ROS, since this sends an entire policy to the robot.
        """
        if t == 0:
            self.inputs = [] #debugging
            self.calculated = []
        self.t = t
        if self.use_ee_sites:
            jacobian_to_use = sitejac
        else:
            jacobian_to_use = eejac

        dX = self.dX 
        dU = self.dU
        if self.restricted_context>0:
            self.restricted_context_list.append(np.r_[prevx, prevu].astype(np.float32))
        #gamma = self.gamma
        self.prevX = prevx
        self.prevU = prevu
        self.inputs.append({'x':x, 'empmu':empmu, 'empsig':empsig, 'prevx':prevx, 'prevu':prevu, 'eejac':eejac, 't':t})
        horizon = min(self.H, self.maxT-t);

        if t==0:
            #with open('net/rec_plane_dump.pkl', 'w') as pklfile:
            #    import cPickle
            #    cPickle.dump(self.dyn_net.layers, pklfile)
            #    print 'Dumped net!!'

            # Execute something for first action.
            self.noise = generate_noise(self.maxT, self.dU, smooth=True, var=3.0, renorm=True)

            H = self.H
            K = np.zeros((H, dU, dX))
            k = np.zeros((H, dU))

            init_noise = 1
            cholPSig = np.tile(np.sqrt(init_noise)*np.eye(dU), [H, 1, 1])
            PSig = np.tile(init_noise*np.eye(dU), [H, 1, 1])
            invPSig = np.tile(1/init_noise*np.eye(dU), [H, 1, 1])

            self.mu = self.dyn_init_mu
            self.sigma = self.dyn_init_sig
            self.xxt = self.sigma + np.outer(self.mu, self.mu)
            self.prev_policy = LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig, cache_kldiv_info=True)
            if self.copy_offline_traj:
                for i in range(t,t+self.prev_policy.T):
                    self.prev_policy.K[i-t] = self.offline_K[i]
                    self.prev_policy.k[i-t] = self.offline_k[i]
            self.rviz_traj[t] == [x, self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]]
            return self.prev_policy


        self.update_emp_dynamics(prevx, prevu, x) # Always update movinv average
        if self.nn_dynamics:
            self.update_nn_dynamics(prevx, prevu, x)

        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            if self.use_kl_constraint:
                # This is LQR with a KL-div constraint
                lgpolicy, eta = self.lqr_kl(horizon, x, self.eta, self.prev_policy, t, jacobian=jacobian_to_use)
                self.eta = eta  # Update eta for the next iteration
            else:
                # This is plain LQR
                lgpolicy, reg_mu, reg_del = self.lqr(t, x, self.prev_policy, reg_mu, reg_del, jacobian=jacobian_to_use)

            # Store traj
            self.prev_policy = lgpolicy
        
        u = self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]
        self.calculated.append({
            'u':u, 'K':np.copy(self.prev_policy.K), 'k':np.copy(self.prev_policy.k), 't':t
            })
        self.rviz_traj[t] = np.r_[x, u]


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

            u = self.offline_K[t].dot(x)+self.offline_k[t]  
            print 'CopyTrajU:', u
            self.rviz_traj[t] = np.r_[x, u]
            return LinearGaussianPolicy(newK, newk, None, None, None)

        # Generate noise - jacobian transpose method
        """
        noise_dir = np.array([0,0,-1])
        noise_vec = eejac[0:3,:].T.dot(noise_dir)
        noise_vec = noise_vec/np.linalg.norm(noise_vec)
        final_noise = noise_vec*np.random.randn(1)
        #noise_covar = np.outer(noise_vec,noise_vec)
        #print 'noise_covar', noise_covar
        #final_noise = noise_covar.dot(np.random.randn(7))
        noise_vec = self.u_noise*final_noise
        """

        # Generate noise - 
        noise_vec = self.u_noise*self.prev_policy.chol_pol_covar[0].dot(self.noise[t])

        self.prev_policy.k[0] += noise_vec
        if self.prev_policy.T > 1:
            self.prev_policy.k[1] += noise_vec

        # Store state and action.
        return self.prev_policy 

    def act(self, x, obs, t, noise, sample):
        """
        Given a state, returns action to take.
        Used by MuJoCo
        """
        LOGGER.debug('T=%d', t)
        #start = time.time()
        dX = self.dX
        dU = self.dU
        gamma = self.gamma #* (0.99 ** t)
        horizon = min(self.H, self.maxT-t);
        jacobian = sample.get(END_EFFECTOR_JACOBIANS, t=t)

        if t==0:
            # Execute something for first action.
            H = self.H
            K = self.offline_K[0:H,:] #np.zeros((H, dU, dX))
            k = self.offline_k[0:H,:] #np.zeros((H, dU))
            k += np.random.randn(H, dU)*0.01
            init_noise = 1
            cholPSig = np.tile(np.sqrt(init_noise)*np.eye(dU), [H, 1, 1])
            PSig = np.tile(init_noise*np.eye(dU), [H, 1, 1])
            invPSig = np.tile(1/init_noise*np.eye(dU), [H, 1, 1])

            # Copy offline traj to use on first timestep
            for i in range(t,t+H):
                K[i-t] = self.offline_K[i]
                k[i-t] = self.offline_k[i]

            U = K[0].dot(x) + k[0] #+ cholPSig[t].dot(np.random.randn(dU));
            U.fill(0.0)

            self.prevU = U;
            self.prevX = x;
            if self.restricted_context:
                self.restricted_context_list.append(np.r_[x, U].astype(np.float32))

            self.prev_policy = LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig, cache_kldiv_info=True)

            #pt = np.r_[self.prevX,self.prevU,x]
            #self.mu = pt
            #self.sigma = np.outer(pt,pt)

            mu0,Phi,m,n0 = self.dynprior.eval(dX, dU, np.r_[x,U,x ].reshape(1, dX+dU+dX))
            self.mu = mu0#self.dyn_init_mu
            self.sigma = Phi#self.dyn_init_sig
            self.xxt = self.sigma + np.outer(self.mu, self.mu)
            return U

        self.update_emp_dynamics(self.prevX, self.prevU, x)
        if self.nn_dynamics:
            self.update_nn_dynamics(self.prevX, self.prevU, x)

        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            if self.use_kl_constraint:
                # This is LQR with a KL-div constraint
                lgpolicy, eta = self.lqr_kl(horizon, x, self.eta, self.prev_policy, t, jacobian=jacobian)
                self.eta = eta  # Update eta for the next iteration
            else:
                # This is plain LQR
                lgpolicy, reg_mu, reg_del = self.lqr(t, x, self.prev_policy, reg_mu, reg_del, jacobian=jacobian)


            # Store traj
            self.prev_policy = lgpolicy

        #TODO: Re-enable noise once this works.
        if self.copy_offline_traj:
            u = self.offline_K[t].dot(x)+self.offline_k[t]
            u += self.u_noise * np.random.randn(7)
        else:
            u = self.prev_policy.K[0].dot(x)+self.prev_policy.k[0]
            u += self.prev_policy.chol_pol_covar[0].dot(self.u_noise*np.random.randn(7))

        LOGGER.debug('Action commanded: %s',u)
        # Store state and action.
        np.clip(u, -10,10)

        self.prevX = x
        self.prevU = u
        if self.restricted_context > 0:
            self.restricted_context_list.append(np.r_[x, u].astype(np.float32))

        if self.nn_dynamics:
            xu = np.r_[x, u].astype(np.float32)
            #self.dyn_net.fwd_single(xu) #Evaluate to update RNN state
            self.dyn_net_state = self.dyn_net.get_recurrent_state()
        #elapsed = time.time()-start
        #print 'Controller Act:', elapsed
        return u

    def get_logprob(self, prevx, prevu, cur_x, sigma, mu):
        """
        Compute log-probability under empirical dynamics
        """
        pt = np.r_[prevx,prevu,cur_x]
        # Adjust gamma based on log probability under current empsig
        empsig = sigma 
        empsig += self.sigreg*np.eye(empsig.shape[0])
        diff = pt-mu
        #try:
        U = sp.linalg.cholesky(empsig)
        #except:
        #    import pdb; pdb.set_trace()

        logdet = 2*sum(np.log(np.diag(U))) #np.log(np.linalg.det(empsig))
        empsig_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(empsig.shape[0]), lower=True, check_finite=False), check_finite=False)
        exp_term = (diff.T.dot(empsig_inv).dot(diff))
        pi_term = (-empsig.shape[0]/2.0)*np.log(2*np.pi)
        logprob = pi_term - 0.5*logdet - 0.5*exp_term
        #Test
        #from scipy.stats import multivariate_normal
        #yy = multivariate_normal.pdf(pt, mean=self.mu, cov=empsig);
        #assert yy == np.exp(logprob)
        print 'Logprob:', logprob
        return logprob


    def update_emp_dynamics(self, prevx, prevu, cur_x):
        """
        Update linear dynamics via moving average
        """

        xu = np.r_[prevx, prevu].astype(np.float32)
        if self.adaptive_gamma:
            try:
                new_logprob = self.get_logprob(prevx, prevu, cur_x, self.sigma, self.mu)
            except Exception as e:
                print 'Warning: bad logprob: ', e
                new_logprob = self.adaptive_gamma_logprob
            #F, f = self.dyn_net.getF(xu)
            #nn_Phi, nn_mu = self.mix_nn_prior(F, f, xu, sigma_x=self.dyn_net.get_sigma_x(), strength=self.nn_prior_strength, use_least_squares=False)
            #nn_logprob = self.get_logprob(prevx, prevu, cur_x, nn_Phi, nn_mu)
            #print 'nn_logprob:', nn_logprob

            if self.adaptive_gamma_logprob is not None:
                prev_logprob = self.adaptive_gamma_logprob

                diff = new_logprob - prev_logprob

                k = 1.02
                # Simple update rule
                if diff > 0:
                    self.gamma *= 1/k
                else:
                    self.gamma *= k

                self.gamma = min(0.1, self.gamma)
                self.gamma = max(0.01, self.gamma)
                self.empsig_N = (1/self.gamma)/100
                print 'New gamma:', self.gamma
                print 'New empsig N:', self.empsig_N
            self.adaptive_gamma_logprob = new_logprob

        pt = np.r_[prevx,prevu,cur_x]
        gamma = self.gamma

        self.mu = self.mu*(1-gamma) + pt*(gamma)

        #pt = pt -self.mu
        self.xxt = self.xxt*(1-gamma) + np.outer(pt,pt)*(gamma)
        self.xxt = 0.5*(self.xxt+self.xxt.T)
        self.sigma = self.xxt - np.outer(self.mu, self.mu)

        # Debug - print log prob
        """
        empsig = self.sigma - np.outer(self.mu, self.mu)
        diff = pt-self.mu
        try:
            logprob = np.log(diff.T.dot(np.linalg.inv(empsig)).dot(diff)) + 2*sum(np.log(np.diag(sp.linalg.cholesky(empsig)))) #np.log(np.linalg.det(empsig))
        except:
            logprob = np.log(diff.T.dot(np.linalg.inv(empsig)).dot(diff)) + np.log(np.linalg.det(empsig))
        self.inputs[self.t]['logprob'] = logprob
        print 'Logprob:', logprob
        """

    def update_nn_dynamics(self, prevx, prevu, cur_x):
        """
        Update neural network dynamics via SGD
        """
        pt = np.r_[prevx, prevu]
        lbl = cur_x
        #self.empsig_N += 0.01
        if self.nn_recurrent:
            if self.restricted_context > 0:
                self.restricted_context_list = self.restricted_context_list[-self.restricted_context:]
                hidden = self.dyn_net.get_init_hidden_state()
                for context in self.restricted_context_list:
                    predicted_x, hidden = self.dyn_net.fwd_single(context,hidden)
                pass
            else:
                predicted_x, hidden = self.dyn_net.fwd_single(pt, self.rnn_hidden_state)
            #print self.rnn_hidden_state
            if self.update_hidden_state:
                self.rnn_hidden_state = hidden
            if self.nn_recurrent_plot:
                self.nn_vis.update(hidden)
            diff = cur_x-predicted_x
            print 'RNN Loss:', diff.T.dot(diff)
        else:
            print 'NN Loss:', self.dyn_net.loss_single(pt, cur_x)
        for i in range(self.nn_update_iters):
            # Lsq use 0.003
            print 'NN Dynamics Loss: %f // Ref:%f' % ( self.dyn_net.train_single(pt, lbl, lr=self.nn_lr, momentum=0.95), 0)#self.dyn_net_ref.obj_vec(pt, lbl))

    def lqr(self, T, x, lgpolicy, reg_mu, reg_del, jacobian=None):
        """
        Plain LQR
        """
        dX = self.dX
        dU = self.dU
        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        horizon = min(self.H, self.maxT-T);

        # Compute forward pass
        cv, Cm, Fd, fc, _, _ = self.estimate_cost(horizon, x, lgpolicy, T, jacobian=jacobian, hist_key='old')

        # Compute optimal policy with short horizon MPC.
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

                Vxx = Vxx #+ reg_mu*np.eye(dX)
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
                    LOGGER.debug('[LQR reg] Increasing mu -> %f', reg_mu)
            elif decrease_mu:
                reg_del = min(1/del0, reg_del/del0)
                delmu = reg_del*reg_mu
                if delmu > min_mu:
                    reg_mu = delmu;
                else:
                    reg_mu = min_mu;
                #LOGGER.debug('[LQR reg] Decreasing mu -> %f', reg_mu)

        policy = LinearGaussianPolicy(K, k, None, cholPSig, None)

        #plot new
        #self.forward(horizon, x, lgpolicy, T, hist_key='new')

        return policy, reg_mu, reg_del        

    def lqr_kl(self, H, x, prev_eta, prev_traj_distr, cur_timestep, jacobian=None):
        """
        LQR with KL-divergence constraint
        """
        # Set KL-divergence step size (epsilon)
        kl_step = 1.0 #self._hyperparams['kl_step'] * step_mult

        #line_search = LineSearch(self._hyperparams['min_eta'])
        min_eta = -np.Inf
        cv, Cm, F, f, new_mu, new_sigma = self.estimate_cost(H, x, prev_traj_distr, cur_timestep, jacobian=jacobian)  # Forward pass using prev traj

        alpha = 0.5 # Step size between 0 and 1
        DGD_MAX_ITER = 1
        THRESHA = 1e-4  # First convergence threshold
        THRESHB = 1e-3  # Second convergence threshold
        MIN_ETA = 1e-4

        for itr in range(DGD_MAX_ITER):
            new_traj_distr, new_eta = self.bwd_kl(H, prev_traj_distr, prev_eta, Cm, cv, F, f)
            _, _, _, _, new_mu, new_sigma = self.estimate_cost(H, x, new_traj_distr, cur_timestep, jacobian=jacobian)  # Forward pass

            # Update min eta if we had a correction after running backward
            if new_eta > prev_eta:
                min_eta = new_eta

            # Compute KL divergence between previous and new distribuition
            kl_div = self.kldiv(new_mu, new_sigma, new_traj_distr,
                                   prev_traj_distr, prev_traj_t_offset=1)
            # Main convergence check - constraint satisfaction
            if (abs(kl_div - kl_step*H) < 0.1*kl_step*H or
                    (itr >= 20 and kl_div < kl_step*H)):
                LOGGER.debug("Iteration %i, KL: %f / %f converged",
                             itr, kl_div, kl_step*H)
                eta = prev_eta  # TODO - should this actually be new_eta? (matlab code does this.)
                break

            # Adjust eta via EG
            #sign = (kl_div - H*kl_step)/ np.abs(kl_div - H*kl_step)
            #eta = np.exp( np.log(new_eta) + alpha*sign)
            #eta = np.exp( np.log(new_eta) + alpha*new_eta*(kl_div-H*kl_step))
            #eta = max(new_eta + alpha*(kl_div-H*kl_step), 1e-6)
            #"""
            step = alpha*new_eta*(kl_div-H*kl_step)
            max_step = 5.0
            step = min(max(-max_step,step),max_step)
            eta = np.exp( np.log(new_eta) + step )
            #"""

            # Convergence check - dual variable change when min_eta hit
            if (abs(prev_eta - eta) < THRESHA and
                        eta == max(min_eta, MIN_ETA)):
                LOGGER.debug("Iteration %i, KL: %f / %f converged (eta limit)",
                             itr, kl_div, kl_step*H)
                break

            # Convergence check - constraint satisfaction, kl not changing much
            if (itr > 2 and abs(kl_div - prev_kl_div) < THRESHB and
                        kl_div < kl_step*H):
                LOGGER.debug("Iteration %i, KL: %f / %f converged (no change)",
                             itr, kl_div, kl_step*H)
                break
            
            prev_kl_div = kl_div
            LOGGER.debug('Iteration %i, KL: %f / %f eta: %f -> %f',
                         itr, kl_div, kl_step*H, prev_eta, eta)
            prev_eta = eta
            #prev_traj_distr = new_traj_distr
        #if kl_div > kl_step*T and abs(kl_div - kl_step*T) > 0.1*kl_step*T:
        #    LOGGER.warning("Final KL divergence after DGD convergence is too high")

        return new_traj_distr, eta

    def kldiv(self, new_mu, new_sigma, new_traj_distr, prev_traj_distr, prev_traj_t_offset=0):
        """Compute KL divergence between new and the previous trajectory
        distributions.

        Args:
            new_mu: T x dX, mean of new trajectory, computed from forward
            new_sigma: T x dX x dX, variance of new trajectory, computed from forward
            new_traj_distr: A linear gaussian policy object, new distribution
            prev_traj_distr: A linear gaussian policy object, previous distribution
        Returns:
            kl_div: KL divergence between new and previous trajectories
        """
        offset = prev_traj_t_offset

        # Constants
        T = new_mu.shape[0]-prev_traj_t_offset
        dU = new_traj_distr.dU

        # Initialize vector of divergences for each time step
        kl_div = np.zeros(T)

        # Step through trajectory
        for t in range(T):
            # Fetch matrices and vectors from trajectory distributions
            mu_t = new_mu[t,:]
            sigma_t = new_sigma[t,:,:]
            K_prev = prev_traj_distr.K[t+offset,:,:]
            K_new = new_traj_distr.K[t,:,:]
            k_prev = prev_traj_distr.k[t+offset,:]
            k_new = new_traj_distr.k[t,:]
            #chol_prev = prev_traj_distr.chol_pol_covar[t,:,:]
            #chol_new = new_traj_distr.chol_pol_covar[t,:,:]

            # Compute log determinants and precision matrices
            logdet_prev = prev_traj_distr.logdet_psig[t+offset] #2*sum(np.log(np.diag(chol_prev)))
            logdet_new = new_traj_distr.logdet_psig[t] #2*sum(np.log(np.diag(chol_new)))
            prc_prev = prev_traj_distr.precision[t+offset] #sp.linalg.solve_triangular(chol_prev,sp.linalg.solve_triangular(chol_prev.T, np.eye(dU), lower=True))
            prc_new = new_traj_distr.precision[t]#sp.linalg.solve_triangular(chol_new,sp.linalg.solve_triangular(chol_new.T, np.eye(dU), lower=True))

            # Construct matrix, vector, and constants
            M_prev = np.r_[np.c_[K_prev.T.dot(prc_prev).dot(K_prev),-K_prev.T.dot(prc_prev)],
                           np.c_[-prc_prev.dot(K_prev), prc_prev]]
            M_new = np.r_[np.c_[K_new.T.dot(prc_new).dot(K_new),-K_new.T.dot(prc_new)],
                           np.c_[-prc_new.dot(K_new), prc_new]]
            v_prev = np.r_[K_prev.T.dot(prc_prev).dot(k_prev),
                           -prc_prev.dot(k_prev)]
            v_new = np.r_[K_new.T.dot(prc_new).dot(k_new),
                           -prc_new.dot(k_new)]
            c_prev = 0.5*k_prev.T.dot(prc_prev).dot(k_prev)
            c_new = 0.5*k_new.T.dot(prc_new).dot(k_new)

            # Compute KL divergence at timestep t
            kl_div[t] = max(0, -0.5*mu_t.T.dot((M_new-M_prev)).dot(mu_t) -
                mu_t.T.dot((v_new-v_prev))  - c_new + c_prev - 0.5*np.sum(sigma_t*(M_new-M_prev)) -
                0.5*logdet_new + 0.5*logdet_prev)

        # Add up divergences across time to get total divergence
        return np.sum(kl_div)

    def bwd_kl(self, horizon, prevpolicy, eta, Cm, cv, Fd, fc):
        dX = self.dX
        dU = self.dU
        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        #horizon = min(self.H, self.maxT-T);

        # Compute optimal action with short horizon MPC.
        K = np.zeros((horizon,dU, dX))
        PSig = np.zeros((horizon, dU, dU))
        cholPSig = np.zeros((horizon, dU, dU))
        invPSig = np.zeros((horizon, dU, dU))
        precision = np.zeros((horizon, dU, dU))
        k = np.zeros((horizon, dU))
        fail = True
        while fail:
            Vxx = np.zeros((dX, dX))
            Vx = np.zeros(dX)
            fail = False
            for t in range(horizon-1, -1, -1):
                F = Fd[t]
                f = fc[t]

                Qtt = Cm[t]/eta
                Qt = cv[t]/eta

                # Add in the trajectory divergence term.
                if t == horizon-1:
                    # Regularization: Add a small diagonal to last Qtt
                    Qtt[iu,iu] += 1e-6*np.eye(dU)
                if t < horizon-1:
                    Qtt = Qtt + np.vstack([
                        np.hstack([prevpolicy.K[t+1].T.dot(prevpolicy.inv_pol_covar[t+1]).dot(prevpolicy.K[t+1]),
                                   -prevpolicy.K[t+1].T.dot(prevpolicy.inv_pol_covar[t+1])]),  # X x (X+U)
                        np.hstack([-prevpolicy.inv_pol_covar[t+1].dot(prevpolicy.K[t+1]),
                                   prevpolicy.inv_pol_covar[t+1]])  # U x (X+U)
                    ])
                    Qt = Qt + np.hstack([prevpolicy.K[t+1].T.dot(prevpolicy.inv_pol_covar[t+1]).dot(
                        prevpolicy.k[t+1]), -prevpolicy.inv_pol_covar[t+1].dot(prevpolicy.k[t+1])])

                Vxx = Vxx 
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
                PSig[t] = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True, check_finite=False), check_finite=False)
                cholPSig[t] = sp.linalg.cholesky(PSig[t], check_finite=False)
                invPSig[t] = np.linalg.inv(PSig[t])

                # Compute value function.
                Vxx = Qtt[ix, ix] + Qtt[ix, iu].dot(K[t])
                Vx = Qt[ix] + Qtt[ix, iu].dot(k[t])
                Vxx = 0.5 * (Vxx + Vxx.T)

            if fail:
                if eta==0 or np.isinf(eta) or np.isnan(eta):
                    raise ValueError("eta is 0/inf/NaN: %f" % eta)
                if eta > 1e5:
                    raise ValueError("Failed to find SPD solution")
                eta = eta*2
                LOGGER.debug('[LQR reg] Increasing eta -> %f', eta)

        policy = LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig, cache_kldiv_info=True)
        return policy, eta

    """
    def forward(self, horizon, x0, lgpolicy, cur_timestep, hist_key=''):
        # Cost + dynamics estimation

        H = horizon

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

        K = lgpolicy.K
        k = lgpolicy.k
        # Perform forward pass.
        for t in range(H):
            #PSig = cholPSig[t].T.dot(cholPSig[t])
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
                    if self.nn_dynamics:
                        pass
                        #self.dyn_net.fwd_single(xu) #Evaluate to update RNN state
                    F[t], f[t], dynsig[t] = self.getdynamics(mu[t-1,ix], mu[t-1,iu], mu[t, ix], cur_timestep+t, cur_action=cur_action);
                #trajsig[t+1,ix,ix] = F[t].dot(trajsig[t]).dot(F[t].T) + dynsig[t]
                mu[t+1,ix] = F[t].dot(mu[t]) + f[t]

        self.fwd_hist[cur_timestep][hist_key] = {'trajmu': mu, 'F': F, 'f': f}
        return mu, trajsig
    """

    def estimate_cost(self, horizon, x0, lgpolicy, cur_timestep, jacobian=None, hist_key=''):
        """
        Returns cost matrices and dynamics via a forward pass
        """
        # Cost + dynamics estimation

        H = horizon

        N = 1

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

        K = lgpolicy.K
        k = lgpolicy.k

        if self.nn_recurrent:
            fwd_rnn_state = self.rnn_hidden_state
        # Perform forward pass.
        for t in range(H):
            PSig = cholPSig[t].T.dot(cholPSig[t])
            trajsig[t] = np.r_[
                                np.c_[trajsig[t,ix,ix], trajsig[t,ix,ix].dot(K[t].T)],
                                np.c_[K[t].dot(trajsig[t,ix,ix]), K[t].dot(trajsig[t,ix,ix]).dot(K[t].T) + PSig]
                             ]
            cur_action = K[t].dot(mu[t,ix])+k[t]
            mu[t] = np.r_[mu[t,ix], cur_action]
            #mu[t] = np.r_[mu[t,ix], np.zeros(dU)]

            # Reuse old dynamics
            if not self.time_varying_dynamics:
                if t==0: 
                    if self.nn_recurrent:
                        F[0], f[0], dynsig[0], fwd_rnn_state = self.getdynamics(self.prevX, self.prevU, x0, cur_timestep, cur_action=cur_action, rnn_state=fwd_rnn_state);
                    else:
                        F[0], f[0], dynsig[0] = self.getdynamics(self.prevX, self.prevU, x0, cur_timestep, cur_action=cur_action);
                F[t] = F[0]
                f[t] = f[0]
                dynsig[t] = dynsig[0]

            if t < H-1:
                # Estimate new dynamics here based on mu
                if self.time_varying_dynamics:
                    if self.nn_recurrent:
                        F[t], f[t], dynsig[t], fwd_rnn_state = self.getdynamics(mu[t-1,ix], mu[t-1,iu], mu[t, ix], cur_timestep+t, cur_action=cur_action, rnn_state=fwd_rnn_state);
                    else:
                        F[t], f[t], dynsig[t] = self.getdynamics(mu[t-1,ix], mu[t-1,iu], mu[t, ix], cur_timestep+t, cur_action=cur_action);
                trajsig[t+1,ix,ix] = F[t].dot(trajsig[t]).dot(F[t].T) + dynsig[t]
                mu[t+1,ix] = F[t].dot(mu[t]) + f[t]

        self.fwd_hist[cur_timestep][hist_key] = {'trajmu': mu, 'F': F, 'f': f, 'empsig':(self.sigma - np.outer(self.mu, self.mu))}

        #TODO: Hardcoded joint indexes
        jacobians = None
        if self.jac_service:
            try:
                #pass
                jacobians = self.jac_service(mu[:,0:7])
            except:
                print 'Warning - jacobians timed out!'
                pass

        self.vis_forward_pass_joints = mu[:,0:7]
        self.vis_forward_ee = mu[:,self.ee_idx] #TODO: Remove hardcoded indices

        #cc = np.zeros((N, H))
        cv = np.zeros((N, H, dT))
        Cm = np.zeros((N, H, dT, dT))

        for n in range(N):
            # Get costs.
            #l,lx,lu,lxx,luu,lux = self.cost.eval(Xs[n],Us[n], cur_timestep);
            #newmu = np.zeros_like(mu[:,iu]) 
            newmu = mu[:,iu]
            if jacobians is not None:
                l,lx,lu,lxx,luu,lux = self.cost.eval(mu[:,ix], newmu, cur_timestep, jac=jacobians);
            else:
                l,lx,lu,lxx,luu,lux = self.cost.eval(mu[:,ix], newmu, cur_timestep, jac=jacobian);
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
        return cv, Cm, F, f, mu, trajsig

    def getdynamics(self, prev_x, prev_u, cur_x, t, cur_action=None, rnn_state=None):
        """
        Returns linear dynamics given state, timestep
        """
        dX = self.dX
        dU = self.dU

        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)

        if self.use_prior_dyn:
            mun = self.dyn_init_mu
            empsig = self.dyn_init_sig
        else:
            mun = self.mu
            empsig = self.sigma
        N = self.empsig_N

        xu = np.r_[cur_x, cur_action].astype(np.float32)
        xux = np.r_[prev_x, prev_u, cur_x]

        if self.nn_dynamics:
            dynsig = np.zeros((dX,dX))
            if self.nn_recurrent:
                F, f, new_rnn_state = self.dyn_net.getF(xu, rnn_state)
            else:
                F, f = self.dyn_net.getF(xu)
            
            nn_Phi, nnf = self.mix_nn_prior(F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
            if self.nn_prior:
                #Mix
                sigma = (N*empsig + nn_Phi)/(N+1)
                sig_chol = sp.linalg.cholesky(sigma[it,it])
                sig_inv = sp.linalg.solve_triangular(sig_chol, sp.linalg.solve_triangular(sig_chol.T, np.eye(dX+dU), lower=True, check_finite=False), check_finite=False)
                #F = (np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip])).T
                F = sig_inv.dot(sigma[it, ip]).T
                f = mun[ip] - F.dot(mun[it])
            if self.nn_recurrent:
                return F, f, dynsig, new_rnn_state
            else:
                return F, f, dynsig
        else:
            if self.gmm_prior:
                mu0,Phi,m,n0 = self.dynprior.eval(dX, dU, xux.reshape(1, dX+dU+dX))
                sigma = (N*empsig + Phi + ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
            elif self.lsq_prior:
                mu0,Phi = (self.dyn_init_mu, self.dyn_init_sig)
                f = self.dyn_init_mu[dX+dU:dX+dU+dX]
                sigma = (N*empsig + Phi)/(N+1) #+ ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
            elif self.gp_prior:
                F, f = self.gp.linearize(xu)
                Phi, mu0 = self.mix_nn_prior(F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
                sigma = (N*empsig + Phi)/(N+1) #+ ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
                print 'GP k:', self.gp.kerneval(xu)
            else:
                sigma = empsig;  # Moving average only
            sigma[it, it] = sigma[it, it] + self.sigreg*np.eye(dX+dU)

            #(np.linalg.pinv(empsig[it, it]).dot(empsig[it, ip])).T
            Fm = (np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip])).T
            fv = mun[ip] - Fm.dot(mun[it]);

            #import pdb; pdb.set_trace()

            #Fms = (np.linalg.pinv(self.sigma[it, it]).dot(self.sigma[it, ip])).T

            dyn_covar = sigma[ip,ip] - Fm.dot(sigma[it,it]).dot(Fm.T)
            dyn_covar = 0.5*(dyn_covar+dyn_covar.T)  # Make symmetric

            return Fm, fv, dyn_covar

    def mix_nn_prior(self, nnF, nnf, xu, sigma_x=None, strength=1.0, use_least_squares=False, full_calculation=False):
        """
        Provide a covariance/bias term for mixing NN with least squares model.
        """
        dX = self.dX
        dU = self.dU

        ix = slice(dX)
        iu = slice(dX, dX+dU)
        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)

        if use_least_squares:
            sigX = self.dyn_init_sig[it,it]
        else:
            sigX = np.eye(dX+dU)*strength

        sigXK = sigX.dot(nnF.T)
        if full_calculation:
            nn_Phi = np.r_[np.c_[sigX, sigXK],
                           np.c_[sigXK.T, nnF.dot(sigX).dot(nnF.T)+sigma_x  ]]
            nn_mu = np.r_[xu, nnF.dot(xu)+nnf]
        else:
            nn_Phi = np.r_[np.c_[sigX, sigXK],
                           np.c_[sigXK.T, np.zeros((dX, dX))  ]]  # Lower right square is unused
            nn_mu = nnf  # Unused

        return nn_Phi, nn_mu

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

