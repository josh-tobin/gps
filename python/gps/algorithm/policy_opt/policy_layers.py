import caffe
import json

class PolicyDataLayer(caffe.Layer):
    """ A data layer for passing data into the network at training time. """

    def setup(self, bottom, top):
        info = json.loads(self.param_str)
        for ind, top_blob in enumerate(info['shape']):
            top[ind].reshape(*top_blob['dim'])
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        # Nothing to do - data will already set externally.
        # TODO - maybe later include way to pass data to this layer
        # and handle batching here.
        pass

    def backward(self, top, propagate_down, bottom):
        pass
