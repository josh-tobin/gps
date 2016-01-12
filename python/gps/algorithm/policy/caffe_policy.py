import caffe
import numpy as np

from gps.algorithm.policy.policy import Policy


class CaffePolicy(Policy):
    """
    A neural network policy implemented in Caffe. The network output is taken to be the mean, and
    Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        test_net: initialized caffe network that can run forward.
        var: Du-dimensional noise variance vector.
    """
    def __init__(self, test_net, var):
        Policy.__init__(self)
        self.net = test_net
        self.chol_pol_covar = np.diag(np.sqrt(var))

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise vector. This will be scaled by the variance.
        """
        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias

        self.net.blobs[self.net.blobs.keys()[0]].data[:] = obs
        action_mean = self.net.forward().values()[0][0]
        u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u
