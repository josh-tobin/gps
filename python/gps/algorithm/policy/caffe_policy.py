import caffe
import numpy as np

from policy import Policy


class CaffePolicy(Policy):
    """
    A neural network policy implemented in Caffe.
    The network output is taken to be the mean, and gaussian noise is added on top of it.

    U = net.forward(obs) + noise

    Args:
        model_proto (string): Filename of model .prototxt file
        caffemodel (string): Filename of .caffemodel file
        var (float vector): Du-dimensional noise variance vector.
        cpu_mode (bool, optional): If true, use the CPU, else use GPU. Default False.
    """
    def __init__(self, model_proto, caffemodel, var, cpu_mode=False):
        Policy.__init__(self)
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        self.net = caffe.Net(model_proto, caffemodel, caffe.TEST)
        self.chol_pol_covar = np.diag(np.sqrt(var))

    def act(self, x, obs, t, noise):
        """
        Return an action for a state.

        Args:
            x: State vector
            obs: Observation vector
            t: timestep
            noise: Action noise vector. This will be scaled by the variance.
        """
        self.net.blobs['data'] = obs
        action_mean = self.net.forward()
        u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u