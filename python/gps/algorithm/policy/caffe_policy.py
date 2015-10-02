import caffe
import numpy as np

from policy import Policy


class CaffePolicy(Policy):
    """
    A neural network policy implemented in Caffe.
    The network output is taken to be the mean, and gaussian noise is added on top of it.

    U = net.forward(obs) + noise
    Where noise ~ N(0, diag(var))

    Args:
        test_net (caffe.Net): initialized caffe network that can run forward
        var (float vector): Du-dimensional noise variance vector.
    """
    def __init__(self, test_net, var):
        Policy.__init__(self)
        self.net = test_net
        self.chol_pol_covar = np.diag(np.sqrt(var))

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.

        Args:
            x: State vector
            obs: Observation vectors
            t: timestep
            noise: Action noise vector. This will be scaled by the variance.
        """
        self.net.blobs['data'] = obs # TODO - how do we know that the data processing/dimensions are correct?
        action_mean = self.net.forward()
        u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u
