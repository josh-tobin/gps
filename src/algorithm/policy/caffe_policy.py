import caffe
import numpy as np

from policy import Policy


class CaffePolicy(Policy):
    def __init__(self, model_proto, caffemodel, var, cpu_mode=False):
        Policy.__init__(self)
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        self.net = caffe.Net(model_proto, caffemodel, caffe.TEST)
        self.chol_pol_covar = np.diag(np.sqrt(var))

    def act(self, x, obs, t, noise=None):
        self.net.blobs['data'] = obs
        action_mean = self.net.forward()
        if noise is None:
            dU = action_mean.shape[0]
            noise = np.random.randn(1, dU)
        u = action_mean + self.chol_pol_covar.T.dot(noise)
        return u