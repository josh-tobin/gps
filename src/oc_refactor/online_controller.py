import numpy as np

from gps.algorithm.policy.policy import Policy
from online_dynamics import *
import config as default_config

CLIP_U = 5  # Min and max torques for each joint

class OnlineController(Policy):
    def __init__(self, config_dict=None):
        super(OnlineController, self).__init__()

        # Take defaults from config        
        for key in default_config:
            setattr(self, key, default_config[key])

        # Override defaults with provided args
        if config_dict is None:
            for key in config_dict:
                setattr(self, key, config_dict[key])

        # Init objects
        self.cost = CostFKOnline(self.eetgt, \
            wu=self.wu, ee_idx=self.ee_idx, jnt_idx=self.jnt_idx, maxT=self.maxT, use_jacobian=True)
        self.prior = ClassRegistry.getClass(self.prior_class)(*self.prior_class_args)
        self.dynamics = OnlineDynamics(gamma, prior, dyn_init_mu, dyn_init_sig)


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
        if t == 0:
            lgpolicy = self.initial_policy()
        else:
            self.dynamics.update(self.prevx, self.prevu, x)
            lgpolicy = self.run_lqr(t, x, self.prev_policy)
        
        u = self.compute_action(lgpolicy, x)
        self.prev_policy = lgpolicy
        self.prevx = x
        self.prevu = u
        return u

    def initial_policy(self):
        """Return LinearGaussianPolicy for timestep 0"""
        H = self.H
        K = np.zeros((H, dU, dX))
        k = np.zeros((H, dU))
        init_noise = 1
        self.gamma = self.init_gamma
        cholPSig = np.tile(np.sqrt(init_noise)*np.eye(dU), [H, 1, 1])
        PSig = np.tile(init_noise*np.eye(dU), [H, 1, 1])
        invPSig = np.tile(1/init_noise*np.eye(dU), [H, 1, 1])
        return LinearGaussianPolicy(K, k, PSig, cholPSig, 
                                    invPSig)

    def compute_action(self, lgpolicy, x, add_noise=True):
        """ 
        Compute dU-dimensional action from a 
        time-varying LG policy's first timestep (and add noise)
        """
        # Only the first timestep of the policy is used
        u = lgpolicy.K[0].dot(x) + lgpolicy.k[0]
        if add_noise:
            u += lgpolicy.chol_pol_covar[0].dot(self.u_noise * np.random.randn(7))
        np.clip(u, -CLIP_U, CLIP_U)
        return u

    def run_lqr(self, t, x, prev_policy):
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
            lgpolicy, reg_mu, reg_del = lqr(self.cost, prev_policy, self.dynamics,
                    horizon, t, x, self.prevx, self.prevu,
                    reg_mu, reg_del, self.del0, self.min_mu, self.lqr_discount,
                    jacobian=None,
                    max_time_varying_horizon=20)
            prev_policy = lgpolicy
        return lgpolicy

