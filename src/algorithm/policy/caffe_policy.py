import caffe
import numpy as np

from policy import Policy


class CaffePolicy(Policy):
    def __init__(self, model_proto, caffemodel, cpu_mode=True):
        Policy.__init__(self)
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
        self.net = caffe.Net(model_proto, caffemodel, caffe.TEST)

    def act(self, x, obs, t):
        self.net.blobs['data'] = obs
        action_mean = self.net.forward()

        #TODO: how to pass in noise?
        #noise = np.random.randn(action_mean.shape[0],1)
        #u = action_mean + self.chol_pol_covar.T.dot(noise)

        return action_mean