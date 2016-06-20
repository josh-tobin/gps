import numpy as np
import logging
from gps.algorithm.policy.policy import Policy
from gps.proto.gps_pb2 import END_EFFECTOR_POINT_JACOBIANS
from online_dynamics import *
from cost_fk_online import CostFKOnline
from helper import iter_module
import config as default_config

LOGGER = logging.getLogger(__name__)
CLIP_U = 5  # Min and max torques for each joint

class OnlineController(Policy):
    def __init__(self, configfiles, config_dict=None):
        super(OnlineController, self).__init__()

        # Take defaults from config
        defaults = apply_config(configfiles)
        for key in defaults:
            setattr(self, key, defaults[key])

        # Override defaults with provided args
        if config_dict is not None:
            for key in config_dict:
                setattr(self, key, config_dict[key])

        # Init objects
        self.cost = CostFKOnline(self.eetgt, \
            wu=self.wu, ee_idx=self.ee_idx, jnt_idx=self.jnt_idx, maxT=self.maxT, use_jacobian=True)
        self.prior = ClassRegistry.getClass(self.prior_class).from_config(*self.prior_class_args, config=self.__dict__)
        self.dynamics = OnlineDynamics(self.gamma, self.prior, self.dyn_init_mu, self.dyn_init_sig, self.dX, self.dU)


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
        if t==0:
            lgpolicy = self.initial_policy()
        else:
            self.dynamics.update(self.prevx, self.prevu, x)
            jacobian = sample.get(END_EFFECTOR_POINT_JACOBIANS, t=t)  #TODO: Jacobian available for only 1 timestep
            lgpolicy = self.run_lqr(t, x, self.prev_policy, jacobian=jacobian)

        u = self.compute_action(lgpolicy, x)
        LOGGER.debug("U=%s", u)
        self.prev_policy = lgpolicy
        self.prevx = x
        self.prevu = u
        return u

    def initial_policy(self):
        """Return LinearGaussianPolicy for timestep 0"""
        dU, dX = self.dU, self.dX
        H = self.H
        K = self.offline_K[:H] #np.zeros((H, dU, dX))
        k = self.offline_k[:H] #np.zeros((H, dU))
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
            lgpolicy, reg_mu, reg_del = lqr(self.cost, prev_policy, self.dynamics,
                    horizon, t, x, self.prevx, self.prevu,
                    reg_mu, reg_del, self.del0, self.min_mu, self.lqr_discount,
                    jacobian=jacobian,
                    max_time_varying_horizon=20)
            prev_policy = lgpolicy
        return lgpolicy

