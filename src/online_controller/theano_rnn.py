import numpy as np
import theano
import cPickle
import logging
import theano.tensor as T

NN_INIT_WT = 0.1

LOGGER = logging.getLogger(__name__)

def hasgpu():
    """
    Returns if device == 'gpu*'
    """
    return theano.config.device.startswith('gpu')

# Functions defined in gpu mode but not cpu mode, or vice versa
# This allows for cpu/gpu independent code
if hasgpu():
    import theano.sandbox.cuda.basic_ops as tcuda
    gpu_host = tcuda.gpu_from_host
else:
    gpu_host = lambda x: x

class Batch(object):
    def __init__(self):
        self.data = {}

    def get_data(self, key):
        return self.data[key]

    def set_data(self, key, value):
        self.data[key] = value

class BaseLayer(object):
    n_instances = 0
    def __init__(self):
        self.layer_id = BaseLayer.n_instances
        BaseLayer.n_instances += 1

    def forward_batch(self, batch):
        raise NotImplementedError()

    def params(self):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__+str(self.layer_id)

class RecurrentLayer(BaseLayer):
    def __init__(self, input_blob, output_blob, clip_blob):
        super(RecurrentLayer, self).__init__()
        self.input_blob = input_blob
        self.output_blob = output_blob
        self.clip_blob = clip_blob

    def forward(self, prev_layer, prev_state):
        raise NotImplementedError()

    def init_recurrent_state(self):
        raise NotImplementedError()

    def forward_batch(self, input_batch):
        input_data = input_batch.get_data(self.input_blob)
        clip_mask = input_batch.get_data(self.clip_blob)

        init_state_data = self.init_recurrent_state().astype(np.float32)
        init_state = theano.shared(init_state_data, name='rnn_state_'+str(self.layer_id))

        def scan_fn(input_layer, clip, prev_state):
            prev_state = clip*prev_state
            next_layer, next_state = self.forward(input_layer, prev_state)
            return next_layer, next_state

        ([layer_out, hidden_states], updates) = theano.scan(fn=scan_fn,
                                      outputs_info=[None, dict(initial=init_state, taps=[-1])],
                                      sequences=[input_data, clip_mask])
        input_batch.set_data(self.output_blob, layer_out)
        #input_batch.set_data('dbg_hidden_state', hidden_states)

class FeedforwardLayer(BaseLayer):
    def __init__(self, input_blobs, output_blob):
        super(FeedforwardLayer, self).__init__()
        self.input_blobs = input_blobs
        self.output_blob = output_blob

    def forward(self, input_batches):
        raise NotImplementedError()

    def forward_batch(self, input_batch):
        inputs = {blob: input_batch.get_data(blob) for blob in self.input_blobs}
        output = self.forward(inputs)
        input_batch.set_data(self.output_blob, output)

class ActivationLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob):
        super(ActivationLayer, self).__init__([input_blob], output_blob)
        self.input_blob = input_blob

    def activation(self, prev_layer):
        raise NotImplementedError()

    def forward(self, input_data):
        prev_layer = input_data[self.input_blob]
        return self.activation(prev_layer)

    def params(self):
        return []

class ReLULayer(ActivationLayer):
    def __init__(self, input_blob, output_blob):
        super(ReLULayer, self).__init__(input_blob, output_blob)

    def activation(self, prev_layer):
        return prev_layer*(prev_layer>0)

class SigmoidLayer(ActivationLayer):
    def __init__(self, input_blob, output_blob):
        super(SigmoidLayer, self).__init__(input_blob, output_blob)

    def activation(self, prev_layer):
        return T.nnet.sigmoid(prev_layer)

class RNNIPLayer(RecurrentLayer):
    def __init__(self, input_blob, output_blob, clip_blob, din, dout):
        super(RNNIPLayer, self).__init__(input_blob, output_blob, clip_blob)    
        self.dout = dout
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_ip_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_ip_wr_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='rnn_ip_b_'+str(self.layer_id))

    def forward(self, prev_layer, prev_state):
        #output = prev_layer.dot(self.wff) + prev_state.dot(self.wr) + self.b
        #output = prev_layer.dot(self.wff) + 0*prev_state.dot(self.wr) + self.b
        output = self.wff.T.dot(prev_layer) + self.b + self.wr.T.dot(prev_state)
        return output, output

    def init_recurrent_state(self):
        return np.zeros(self.dout)

    def params(self):
        #return [self.wff, self.wr, self.b]
        return [self.wff, self.b]

class FFIPLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob, din, dout):
        super(FFIPLayer, self).__init__([input_blob], output_blob)  
        self.input_blob = input_blob
        self.w = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ff_ip_w_'+str(self.layer_id))
        self.b = theano.shared(NN_INIT_WT*np.random.randn(dout).astype(np.float32), name='ff_ip_b_'+str(self.layer_id))

    def forward(self, input_data):
        prev_layer = input_data[self.input_blob]
        return prev_layer.dot(self.w)+self.b

    def params(self):
        return [self.w, self.b]

class AccelLayer(FeedforwardLayer):
    """ Acceleration Layer """
    def __init__(self, data_blob, accel_blob, output_blob):
        super(AccelLayer, self).__init__([data_blob, accel_blob], output_blob)  
        self.data_blob = data_blob
        self.accel_blob = accel_blob

        self.idxpos = slice(0,7)
        self.idxvel = slice(7,14)
        self.idxeepos = slice(14,23)
        self.idxeevel = slice(23,32)
        self.idxu = slice(32,39)

        self.construct_forward_matrices()

    def construct_forward_matrices(self):
        t = 0.05
        forward_mat = np.zeros((39, 32))
        forward_mat[self.idxpos, self.idxpos] = np.eye(7)
        forward_mat[self.idxvel, self.idxvel] = np.eye(7)
        forward_mat[self.idxvel, self.idxpos] = t*np.eye(7)
        forward_mat[self.idxeepos, self.idxeepos] = np.eye(9)
        forward_mat[self.idxeevel, self.idxeevel] = np.eye(9)
        forward_mat[self.idxeevel, self.idxeepos] = t*np.eye(9)
        self.forward_mat = theano.shared(forward_mat.astype(np.float32), name="AccLayer_forward_mat_"+str(self.layer_id))

        jnt_mat = np.zeros((16, 32))
        jnt_mat[:7, self.idxpos] = t*t*np.eye(7)
        jnt_mat[:7, self.idxvel] = t*np.eye(7)
        jnt_mat[7:16, self.idxeepos] = t*t*np.eye(9)
        jnt_mat[7:16, self.idxeevel] = t*np.eye(9)
        self.jnt_mat = theano.shared(jnt_mat.astype(np.float32), name="AccLayer_acc_mat_"+str(self.layer_id))

    def forward(self, input_data):
        acc = input_data[self.accel_blob]
        jnts_data = input_data[self.data_blob]
        return jnts_data.dot(self.forward_mat) + acc.dot(self.acc_mat)

class SquaredLoss(object):
    def __init__(self, predict_blob, lbl_blob, wt=None):
        super(SquaredLoss, self).__init__()
        self.wt = wt
        self.predict_blob = predict_blob
        self.lbl_blob = lbl_blob

    def loss(self, labels, predictions):
        diff = labels-predictions
        if self.wt is not None:
            diff = diff*self.wt
        loss = T.sum(diff*diff)/diff.shape[0]
        return loss

    def forward_batch(self, batch):
        lbl = batch.get_data(self.lbl_blob)
        pred = batch.get_data(self.predict_blob)
        obj = self.loss(lbl, pred)
        return obj

def train_gd_momentum(obj, params, args, scl=1.0, weight_decay=0.0):
    obj = obj
    scl = float(scl)
    gradients = T.grad(obj, params)
    eta = T.scalar('lr')
    momentum = T.scalar('momentum')
    momentums = [theano.shared(np.copy(param.get_value())) for param in params]
    updates = []
    for i in range(len(gradients)):
        update_gradient = (gradients[i])+momentum*momentums[i]+weight_decay*params[i]
        updates.append((params[i], gpu_host(params[i]-(eta/scl)*update_gradient)))
        updates.append((momentums[i], gpu_host(update_gradient)))
    train = theano.function(
        inputs=args+[eta, momentum],
        outputs=[obj],
        updates=updates
    )
    return train

class Network(object):
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params())

    def symbolic_forward(self):
        batch = Batch()
        for input_var in self.inputs:
            batch.set_data(input_var.name, input_var)

        for layer in self.layers:
            layer.forward_batch(batch)
        obj = self.loss.forward_batch(batch)
        return obj

    def set_net_inputs(self, inputs):
        self.inputs = inputs

    def get_train_function(self, objective):
        return train_gd_momentum(objective, self.params, self.inputs)

    def get_loss_function(self, objective):
        return theano.function(inputs=self.inputs, outputs=[objective])

    def taylor_expansion(self, x):
        raise NotImplementedError()

    def get_hidden_state(self):
        raise NotImplementedError()

    def set_hidden_state(self, state):
        raise NotImplementedError()

    def __getstate__(self):  # For pickling
        return (self.layers, self.loss)

    def __setstate__(self, state):  # For pickling
        self.__init__(state[0], state[1])

    def pickle(self, fname):
        LOGGER.debug('Dumping network to: %s', fname)
        with open(fname, 'w') as pklfile:
            cPickle.dump(self, pklfile)

def unpickle_net(fname):
    LOGGER.debug('Loading network from: %s', fname)
    with open(fname, 'r') as pklfile:
        net = cPickle.load(pklfile)
    return net

def rnntest():
    np.random.seed(123)
    logging.basicConfig(level=logging.DEBUG)

    bsize = 50
    N = 500

    data = np.random.randn(N,40).astype(np.float32)
    K = np.random.randn(40,20).astype(np.float32)
    label = data.dot(K)
    clip = np.zeros((N,)).astype(np.float32)

    ip1 = RNNIPLayer('data', 'ip1', 'clip', 40, 20) 
    loss = SquaredLoss('ip1', 'lbl')
    net = Network([ip1], loss)
    #net = unpickle_net('test.pkl')

    net.set_net_inputs([T.matrix('data'), T.matrix('lbl'), T.vector('clip')])
    obj = net.symbolic_forward()
    train_gd = net.get_train_function(obj)
    total_obj = net.get_loss_function(obj)


    lr = 5e-3/bsize
    lr_schedule = {
        400000: 0.2,
        800000: 0.2,
    }
    epochs = 0
    for i in range(100000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            perm = np.random.permutation(N)
            data = data[perm]
            label = label[perm]
            clip = clip[perm]
            continue
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _clip = clip[bstart:bend]
        objval = train_gd(_data, _label, _clip, lr, 0.90)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 500 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            #import pdb; pdb.set_trace()
        if i % 10000 == 0:
            pass
            #if i>0:
            #    net.pickle('test.pkl')
            #total_err = total_obj(data, label, clip)
            #print 'Total train error:', total_err
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    rnntest()
