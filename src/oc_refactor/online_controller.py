import numpy as np
import logging
from gps.algorithm.policy.policy import Policy
from gps.algorithm.policy.tf_policy import TfPolicy
from gps.proto.gps_pb2 import END_EFFECTOR_POINT_JACOBIANS
from online_dynamics import *
from cost_fk_online import CostFKOnline
from helper import iter_module

LOGGER = logging.getLogger(__name__)
CLIP_U = 2.5

class OnlineController(TfPolicy):
    def __init__(self, configfiles, config_dict=None):
        super(TfPolicy, self).__init__()

        defaults = apply_config(configfiles)
        for key in defaults:
            setattr(self, key, defaults[key])
        if config_dict is not None:
            for key in config_dict:
                setattr(self, key, config_dict[key])

        # Init objects
        self.cost = CostFKOnline(
                self.eetgt,
                wu=self.wu,
                ee_idx=self.ee_idx,
                jnt_idx=self.jnt_idx,
                maxT=self.maxT,
                use_jacobian=self.use_jacobian,
                l1=self.l1,
                l2=self.l2,
                ramp_option=self.ramp_option,
                wp=self.wp,
                final_penalty=self.final_penalty,

        )
        self.prior = ClassRegistry.getClass(self.prior_class).from_config(*self.prior_class_args, config=self.__dict__)
        self.dynamics = ClassRegistry.getClass(self.dynamics_class).from_config(self.prior, config=self.__dict__)
        #self.offline_controller 

        self.prevx = None
        self.prevu = None
        self.prev_policy = None
        self.u_history = []

    def act(self, x, obs, t, noise=None, sample=None):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        LOGGER.debug("Timestep=%d", t)
        print 'Timestep=%d, err=%f'%(t, 
	    np.sum(np.abs(self.eetgt-obs[14:23])))
	if x is None:
            fb = obs[:self.dX]
        else:
            fb = x
        if t == 0 or self.prevx is None:
            lgpolicy = self.initial_policy()
        else:
            self.dynamics.update(self.prevx, self.prevu, fb)
            if sample is not None:
                jacobian = sample.get(END_EFFECTOR_POINT_JACOBIANS, t=t)
            else:
                jacobian = obs[self.dX:].reshape([-1,self.dU])
            lgpolicy = self.run_lqr(t, fb, self.prev_policy, 
                    jacobian=jacobian)
        u = self.compute_action(lgpolicy, fb, add_noise=True)
        LOGGER.debug("U=%s", u)
        self.prev_policy = lgpolicy
        self.prevx = fb
        self.prevu = u
        self.u_history.append(u)

        return u

    def initial_policy(self):
        dU, dX = self.dU, self.dX
        H = self.H
        K = self.offline_K[:H]
        k = self.offline_k[:H]
        #init_noise=self.init_noise
        init_noise =  1
        self.gamma = self.init_gamma
        cholPSig = np.tile(np.sqrt(init_noise)*np.eye(dU), [H,1,1])
        PSig = np.tile(np.sqrt(init_noise)*np.eye(dU), [H,1,1])
        invPSig = np.tile(1/init_noise*np.eye(dU), [H,1,1])
        return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)


    def compute_action(self, lgpolicy, x, add_noise=True):
        """
        Compute dU-dimensional action from a
        time-varying LG policy's first timestep (and add noise)
        """
        # Only the first timestep of the policy is used
	u = lgpolicy.K[0].dot(x) + lgpolicy.k[0]
        if add_noise:
            u += lgpolicy.chol_pol_covar[0].dot(self.u_noise * np.random.randn(7))
        u = np.clip(u, -CLIP_U, CLIP_U)
        return u

    
    def run_lqr(self, t, x, prev_policy, jacobian=None):
        """
        Compute a new policy given new state

        Returns:
            LinearGaussianPolicy: An updated policy
        """
        horizon = min(self.H, self.maxT - t)
        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            # This is plain LQR
            if self.use_offline_value:
                Vxx = self.offline_Vxx
                Vx = self.offline_Vx
            else:
                Vxx = None
                Vx = None
            lgpolicy, reg_mu, reg_del = lqr(
		self.cost, prev_policy, self.dynamics,
                horizon, t, x, self.prevx, self.prevu,
                reg_mu, reg_del, self.del0, self.min_mu,
		self.lqr_discount, jacobian=jacobian,
                max_time_varying_horizon=horizon,
                offline_Vxx=Vxx, offline_Vx=Vx)
            prev_policy = lgpolicy
        return lgpolicy

