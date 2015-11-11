import numpy as np

from gps.algorithm.policy_opt.policy_opt import PolicyOpt


class PolicyOptDummy(PolicyOpt):
    """
    Not actual policy optimization. Just a dummy filler class.
    """

    def __init__(self, hyperparams, dO, dU):
        self.dO = dO
        self.dU = dU

    def update(self, obs, mu, prc, wt):
        pass

    def prob(self, obs):
        N, T = obs.shape[:2]
        dU = self.dU
        return np.zeros((N, T, dU)), np.zeros((N, T, dU, dU)), \
                np.zeros((N, T, dU, dU)), np.zeros((N, T))
