import numpy as np

from gps.algorithm.policy.policy import Policy
from online_dynamics import *

CLIP_U = 5  # Min and max torques for each joint

class OnlineController(Policy):
    def __init__(self):
        #TODO(justin): Find a better way to feed in a long list of params
        prior=None
        self.dynamics = OnlineDynamics(0.5, prior)
        self.cost = None
        self.dX = None
        self.dU = None

        # LQR
        self.LQR_iter = 1  # Number of LQR iterations to take
        self.min_mu = 1e-6  # LQR regularization
        self.del0 = 2  # LQR regularization
        self.lqr_discount = 0.9

        # Horizon
        self.H = 12
        # Timesteps
        self.maxT = 100

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
            u = self.initial_action()
        else:
            self.dynamics.update(self.prevx, self.prevu, x)
            self.prev_policy = lgpolicy = self.run_lqr(t, x, self.prev_policy)
            u = self.compute_action(lgpolicy, x)

        self.prevx = x
        self.prevu = u
        return u

    def initial_action(self):
        raise NotImplementedError()

    def compute_action(self, lgpolicy, x):
        u = lgpolicy.K[0].dot(x) + lgpolicy.k[0]
        u += lgpolicy.chol_pol_covar[0].dot(self.u_noise * np.random.randn(7))
        # Store state and action.
        np.clip(u, -CLIP_U, CLIP_U)
        return u

    def run_lqr(self, t, x, prev_policy):
        horizon = min(self.H, self.maxT - t)
        reg_mu = self.min_mu
        reg_del = self.del0
        for _ in range(self.LQR_iter):
            # This is plain LQR
            lgpolicy, reg_mu, reg_del = lqr(self.cost, prev_policy, self.dynamics,
                    horizon, self.dX, self.dU, t, x, self.prevx, self.prevu,
                    reg_mu, reg_del, self.del0, self.min_mu, self.lqr_discount,
                    jacobian=None,
                    max_time_varying_horizon=20)
            prev_policy = lgpolicy
        return lgpolicy

