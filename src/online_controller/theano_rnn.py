import numpy as np
import theano
import cPickle
import logging
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

NN_INIT_WT = 0.1
STAGE_TRAIN = 'train'
STAGE_TEST = 'test'

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

    def set_state_value(self, state): #Recurrent state
        # Default: Do nothing
        pass

    def get_state_value(self): #Recurrent state
        return None

    def update(self, stage=STAGE_TRAIN):
        pass

    def __str__(self):
        return self.__class__.__name__+str(self.layer_id)

class RecurrentLayer(BaseLayer):
    def __init__(self, input_blob, output_blob, clip_blob):
        super(RecurrentLayer, self).__init__()
        self.__input_blob = input_blob
        self.__output_blob = output_blob
        self.__clip_blob = clip_blob

    def forward(self, prev_layer, prev_state):
        raise NotImplementedError()

    def init_recurrent_state(self):
        raise NotImplementedError()

    def set_state_value(self, state):
        raise NotImplementedError()

    def get_state_value(self):
        raise NotImplementedError()

    def get_state_var(self):
        raise NotImplementedError()

    def forward_batch(self, input_batch):
        input_data = input_batch.get_data(self.__input_blob)
        clip_mask = input_batch.get_data(self.__clip_blob)

        #init_state_data = self.init_recurrent_state().astype(np.float32)
        #hidden_state = theano.shared(init_state_data, name='rnn_state_'+str(self.layer_id))
        hidden_state = self.get_state_var()

        def scan_fn(input_layer, clip, prev_state):
            prev_state = clip*prev_state
            next_layer, next_state = self.forward(input_layer, prev_state)
            return next_layer, next_state

        ([layer_out, hidden_states], updates) = theano.scan(fn=scan_fn,
                                      outputs_info=[None, dict(initial=hidden_state, taps=[-1])],
                                      sequences=[input_data, clip_mask])
        #theano.printing.debugprint(hidden_states)
        #theano.printing.debugprint(layer_out)
        input_batch.set_data(self.__output_blob, layer_out)
        #input_batch.set_data('dbg_hidden_state', hidden_states)
        return [(hidden_state,  hidden_states[-1])]

class FeedforwardLayer(BaseLayer):
    def __init__(self, input_blobs, output_blobs):
        super(FeedforwardLayer, self).__init__()
        if isinstance(input_blobs, basestring):
            input_blobs = [input_blobs]
        self.__input_blobs = input_blobs
        self.__output_blobs = output_blobs

    def forward(self, input_batches, stage=STAGE_TRAIN):
        raise NotImplementedError()

    def forward_batch(self, input_batch):
        inputs = {blob: input_batch.get_data(blob) for blob in self.__input_blobs}
        outputs = self.forward(inputs)
        if not isinstance(self.__output_blobs, basestring):
            assert len(outputs) == len(self.__output_blobs)
            for i in range(len(outputs)):
                input_batch.set_data(self.__output_blobs[i], outputs[i])
        else:
            input_batch.set_data(self.__output_blobs, outputs)
        return []

class ActivationLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob):
        super(ActivationLayer, self).__init__([input_blob], output_blob)
        self.__input_blob = input_blob

    def activation(self, prev_layer):
        raise NotImplementedError()

    def forward(self, input_data):
        prev_layer = input_data[self.__input_blob]
        return self.activation(prev_layer)

    def params(self):
        return []

ACTIVATION_DICT = {
    'softplus': T.nnet.softplus,
    'tanh': T.tanh
}

class ReLULayer(ActivationLayer):
    def __init__(self, input_blob, output_blob):
        super(ReLULayer, self).__init__(input_blob, output_blob)

    def activation(self, prev_layer):
        return prev_layer*(prev_layer>0)

class SoftplusLayer(ActivationLayer):
    def __init__(self, input_blob, output_blob):
        super(SoftplusLayer, self).__init__(input_blob, output_blob)

    def activation(self, prev_layer):
        return T.nnet.softplus(prev_layer)

class SigmoidLayer(ActivationLayer):
    def __init__(self, input_blob, output_blob):
        super(SigmoidLayer, self).__init__(input_blob, output_blob)

    def activation(self, prev_layer):
        return T.nnet.sigmoid(prev_layer)

class TanhLayer(ActivationLayer):
    def __init__(self, input_blob, output_blob):
        super(TanhLayer, self).__init__(input_blob, output_blob)

    def activation(self, prev_layer):
        return T.nnet.tanh(prev_layer)

class RNNIPLayer(RecurrentLayer):
    def __init__(self, input_blob, output_blob, clip_blob, din, dout, activation=None):
        super(RNNIPLayer, self).__init__(input_blob, output_blob, clip_blob)    
        self.input_blob = input_blob
        self.output_blob = output_blob
        self.din = din
        self.dout = dout
        self.activation = activation
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_ip_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_ip_wr_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='rnn_ip_b_'+str(self.layer_id))
        self.state = theano.shared(np.zeros(dout).astype(np.float32), name='rnn_ip_state_'+str(self.layer_id))

    def forward(self, prev_layer, prev_state):
        #output = prev_layer.dot(self.wff) + prev_state.dot(self.wr) + self.b
        #output = prev_layer.dot(self.wff) + 0*prev_state.dot(self.wr) + self.b
        output = self.wff.T.dot(prev_layer) + self.b + self.wr.T.dot(prev_state)
        if self.activation:
            output = ACTIVATION_DICT[self.activation](output)
        return output, output

    def init_recurrent_state(self):
        return np.zeros(self.dout)

    def set_state_value(self, state):
        self.state.set_value(state)

    def get_state_value(self):
        return self.state.get_value()

    def get_state_var(self):
        return self.state

    def params(self):
        return [self.wff, self.wr, self.b]

    def to_feedforward_test(self):
        #self.input_blob = self._RecurrentLayer__input_blob
        #self.output_blob = self._RecurrentLayer__output_blob
        #self.din = self.wff.get_value().shape[0]
        #self.dout = self.wff.get_value().shape[1]
        new_layer = FF_RNNIPLayer(self.input_blob, self.output_blob, self.din, self.dout, activation=self.activation)
        new_layer.wff.set_value(self.wff.get_value())
        new_layer.wr.set_value(self.wr.get_value())
        new_layer.b.set_value(self.b.get_value())
        return new_layer


class FF_RNNHackLayer(FeedforwardLayer):
    def __init__(self, input_blobs, output_blobs):
        super(FF_RNNHackLayer, self).__init__(input_blobs, output_blobs)

    def hidden_state_io_blobs(self):
        raise NotImplementedError()

    def init_recurrent_state(self):
        raise NotImplementedError()
    
class FF_RNNIPLayer(FF_RNNHackLayer):
    """ 
    A feedforward version of RNNIPLayer that accepts a hidden state as an input instead of storing it implicitly.
    Used for speed reasons, since the scan implementation of recurrent layers is slow.
    """
    n_instances = 0
    def __init__(self, input_blob, output_blob, din, dout, activation=None):
        FF_RNNIPLayer.n_instances += 1
        self.ffrnn_layer_id = FF_RNNIPLayer.n_instances
        self.hidden_state_blob = 'ffrnn_ip_hidden_'+str(self.ffrnn_layer_id)
        self.output_state_blob = self.hidden_state_blob+'_out'
        super(FF_RNNIPLayer, self).__init__([input_blob, self.hidden_state_blob], [output_blob, self.output_state_blob])    
        self.dout = dout
        self.activation = activation
        self.input_blob = input_blob
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ffrnn_ip_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='ffrnn_ip_wr_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='ffrnn_ip_b_'+str(self.layer_id))

    def forward(self, input_data):
        prev_layer = input_data[self.input_blob]
        prev_state = input_data[self.hidden_state_blob]
        output = prev_layer.dot(self.wff) + self.b + prev_state.dot(self.wr)
        if self.activation:
            output = ACTIVATION_DICT[self.activation](output)
        return output, output
    
    def hidden_state_io_blobs(self):
        return self.hidden_state_blob, self.output_state_blob

    def init_recurrent_state(self):
        return np.zeros(self.dout) #self.hidden_state_blob, self.output_state_blob

    def params(self):
        return [self.wff, self.wr, self.b]


class SimpGateLayer(RecurrentLayer):
    def __init__(self, input_blob, output_blob, clip_blob, din, dout):
        super(SimpGateLayer, self).__init__(input_blob, output_blob, clip_blob)    
        self.input_blob = input_blob
        self.output_blob = output_blob
        self.din = din
        self.dout = dout
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wr_'+str(self.layer_id))
        self.wreset_in = theano.shared(0.1*NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wresetin_'+str(self.layer_id))
        self.wreset_hidden = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wresethidden_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='rnn_g_b_'+str(self.layer_id))
        self.state = theano.shared(np.zeros(dout).astype(np.float32), name='rnn_g_state_'+str(self.layer_id))

    def forward(self, prev_layer, prev_state):
        reset_value = T.nnet.sigmoid(self.wreset_in.T.dot(prev_layer)+self.wreset_hidden.T.dot(prev_state))
        new_state = T.tanh(self.wff.T.dot(prev_layer) + self.wr.T.dot(reset_value*prev_state))
        #interp = gate_value*new_state + (1-gate_value)*prev_state  # Linearly interpolate btwn new and old state
        return new_state, new_state

    def init_recurrent_state(self):
        return np.zeros(self.dout)

    def set_state_value(self, state):
        self.state.set_value(state)

    def get_state_value(self):
        return self.state.get_value()

    def get_state_var(self):
        return self.state

    def params(self):
        return [self.wff, self.wr, self.wreset_in, self.wreset_hidden]

    def to_feedforward_test(self):
        new_layer = FF_SimpGateLayer(self.input_blob, self.output_blob, self.din, self.dout)
        new_layer.wff.set_value(self.wff.get_value())
        new_layer.wr.set_value(self.wr.get_value())
        new_layer.wreset_in.set_value(self.wreset_in.get_value())
        new_layer.wreset_hidden.set_value(self.wreset_hidden.get_value())

        new_layer.b.set_value(self.b.get_value())
        return new_layer

class FF_SimpGateLayer(FF_RNNHackLayer):
    """ 
    A feedforward version of RNNIPLayer that accepts a hidden state as an input instead of storing it implicitly.
    Used for speed reasons, since the scan implementation of recurrent layers is slow.
    """
    n_instances = 0
    def __init__(self, input_blob, output_blob, din, dout):
        FF_SimpGateLayer.n_instances += 1
        self.ffrnn_layer_id = FF_SimpGateLayer.n_instances
        self.hidden_state_blob = 'ffrnn_g_hidden_'+str(self.ffrnn_layer_id)
        self.output_state_blob = self.hidden_state_blob+'_out'
        super(FF_SimpGateLayer, self).__init__([input_blob, self.hidden_state_blob], [output_blob, self.output_state_blob])    
        self.din = din
        self.dout = dout
        self.input_blob = input_blob
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ffrnn_g_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='ffrnn_g_wr_'+str(self.layer_id))
        self.wreset_in = theano.shared(0.1*NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ffrnn_g_wresetin_'+str(self.layer_id))
        self.wreset_hidden = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='ffrnn_g_wresethidden_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='ffrnn_g_b_'+str(self.layer_id))

    def forward(self, input_data):
        prev_layer = input_data[self.input_blob]
        prev_state = input_data[self.hidden_state_blob]

        reset_value = T.nnet.sigmoid(prev_layer.dot(self.wreset_in)+prev_state.dot(self.wreset_hidden))
        new_state = T.tanh(prev_layer.dot(self.wff) + (reset_value*prev_state).dot(self.wr) + self.b)
        #interp = gate_value*new_state + (1-gate_value)*prev_state  # Linearly interpolate btwn new and old state
        return new_state, new_state
    
    def hidden_state_io_blobs(self):
        return self.hidden_state_blob, self.output_state_blob

    def init_recurrent_state(self):
        return np.zeros(self.dout)

    def params(self):
        return [self.wff, self.wr, self.wreset_in, self.wreset_hidden, self.b]

class GRULayer(RecurrentLayer):
    def __init__(self, input_blob, output_blob, clip_blob, din, dout):
        super(GRULayer, self).__init__(input_blob, output_blob, clip_blob)    
        self.input_blob = input_blob
        self.output_blob = output_blob
        self.din = din
        self.dout = dout
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wr_'+str(self.layer_id))
        self.wgate_in = theano.shared(0.1*NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wgatein_'+str(self.layer_id))
        self.wgate_hidden = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wgatehidden_'+str(self.layer_id))
        self.wreset_in = theano.shared(0.1*NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wresetin_'+str(self.layer_id))
        self.wreset_hidden = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wresethidden_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='rnn_g_b_'+str(self.layer_id))
        self.state = theano.shared(np.zeros(dout).astype(np.float32), name='rnn_g_state_'+str(self.layer_id))

    def forward(self, prev_layer, prev_state):
        gate_value = T.nnet.sigmoid(self.wgate_in.T.dot(prev_layer)+self.wgate_hidden.T.dot(prev_state))
        reset_value = T.nnet.sigmoid(self.wreset_in.T.dot(prev_layer)+self.wreset_hidden.T.dot(prev_state))
        new_state = T.tanh(self.wff.T.dot(prev_layer) + self.wr.T.dot(reset_value*prev_state) + self.b)
        interp = gate_value*new_state + (1-gate_value)*prev_state  # Linearly interpolate btwn new and old state
        return interp, interp

    def init_recurrent_state(self):
        return np.zeros(self.dout)

    def set_state_value(self, state):
        self.state.set_value(state)

    def get_state_value(self):
        return self.state.get_value()

    def get_state_var(self):
        return self.state

    def params(self):
        return [self.wff, self.wr, self.wreset_in, self.wreset_hidden, self.wgate_in, self.wgate_hidden, self.b]

    def to_feedforward_test(self):
        new_layer = FF_GRULayer(self.input_blob, self.output_blob, self.din, self.dout)
        new_layer.wff.set_value(self.wff.get_value())
        new_layer.wr.set_value(self.wr.get_value())
        new_layer.wgate_in.set_value(self.wgate_in.get_value())
        new_layer.wgate_hidden.set_value(self.wgate_hidden.get_value())

        #AAA
        new_layer.wreset_in.set_value(self.wreset_in.get_value())
        new_layer.wreset_hidden.set_value(self.wreset_hidden.get_value())

        new_layer.b.set_value(self.b.get_value())
        return new_layer

class FF_GRULayer(FF_RNNHackLayer):
    """ 
    A feedforward version of RNNIPLayer that accepts a hidden state as an input instead of storing it implicitly.
    Used for speed reasons, since the scan implementation of recurrent layers is slow.
    """
    n_instances = 0
    def __init__(self, input_blob, output_blob, din, dout):
        FF_GRULayer.n_instances += 1
        self.ffrnn_layer_id = FF_GRULayer.n_instances
        self.hidden_state_blob = 'ffrnn_g_hidden_'+str(self.ffrnn_layer_id)
        self.output_state_blob = self.hidden_state_blob+'_out'
        super(FF_GRULayer, self).__init__([input_blob, self.hidden_state_blob], [output_blob, self.output_state_blob])    
        self.din = din
        self.dout = dout
        self.input_blob = input_blob
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ffrnn_g_wff_'+str(self.layer_id))
        self.wr = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='ffrnn_g_wr_'+str(self.layer_id))
        self.wgate_in = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ffrnn_g_wgatein_'+str(self.layer_id))
        self.wgate_hidden = theano.shared(NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='ffrnn_g_wgatehidden_'+str(self.layer_id))
        self.wreset_in = theano.shared(0.1*NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ffrnn_g_wresetin_'+str(self.layer_id))
        self.wreset_hidden = theano.shared(0.1*NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='ffrnn_g_wresethidden_'+str(self.layer_id))

        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='ffrnn_g_b_'+str(self.layer_id))

    def forward(self, input_data):
        prev_layer = input_data[self.input_blob]
        prev_state = input_data[self.hidden_state_blob]

        gate_value = T.nnet.sigmoid(prev_layer.dot(self.wgate_in)+prev_state.dot(self.wgate_hidden))
        reset_value = T.nnet.sigmoid(prev_layer.dot(self.wreset_in)+prev_state.dot(self.wreset_hidden))
        new_state = T.tanh(prev_layer.dot(self.wff) + (reset_value*prev_state).dot(self.wr) + self.b)
        interp = gate_value*new_state + (1-gate_value)*prev_state  # Linearly interpolate btwn new and old state

        return interp, interp
    
    def hidden_state_io_blobs(self):
        return self.hidden_state_blob, self.output_state_blob

    def init_recurrent_state(self):
        return np.zeros(self.dout)

    def params(self):
        return [self.wff, self.wr, self.wreset_in, self.wreset_hidden, self.wgate_in, self.wgate_hidden, self.b]


class FFIPLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob, din, dout):
        super(FFIPLayer, self).__init__(input_blob, output_blob)  
        self.input_blob = input_blob
        self.w = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='ff_ip_w_'+str(self.layer_id))
        self.b = theano.shared(NN_INIT_WT*np.random.randn(dout).astype(np.float32), name='ff_ip_b_'+str(self.layer_id))

    def forward(self, input_data, stage=STAGE_TRAIN):
        prev_layer = input_data[self.input_blob]
        return prev_layer.dot(self.w)+self.b

    def params(self):
        return [self.w, self.b]

class DropoutLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob, din, rngseed=10, p=0.5):
        super(DropoutLayer, self).__init__(input_blob, output_blob)  
        self.input_blob = input_blob
        self.din = din
        self.p = p
        self.rand = theano.shared(np.zeros(din).astype(np.float32), name='drop_rand_'+str(self.layer_id))
    
    def update(self, stage=STAGE_TRAIN):
        if stage == STAGE_TRAIN:
            self.rand.set_value(np.random.binomial(1, self.p, self.din).astype(np.float32))
        else:
            p = self.rand.get_value()
            p.fill(self.p)
            self.rand.set_value(p)

    def forward(self, input_data, stage=STAGE_TRAIN):
        prev_layer = input_data[self.input_blob]
        return prev_layer * self.rand

    def params(self):
        return []


class NormalizeLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob):
        super(NormalizeLayer, self).__init__(input_blob, output_blob)  
        self.input_blob = input_blob
        self.mean = None
        self.sig = None

    def forward(self, input_data):
        prev_layer = input_data[self.input_blob]
        return prev_layer*self.sig + self.mean

    def generate_weights(self, data):
        self.mean = theano.shared(np.mean(data, axis=0).astype(np.float32), name='normalize_mean_'+str(self.layer_id))
        data = data-self.mean.get_value()
        sig = np.std(data, axis=0)
        self.sig = theano.shared(sig.astype(np.float32), name='normalize_sig_'+str(self.layer_id))

    def params(self):
        return []

class AccelLayer(FeedforwardLayer):
    """ Mix known dynamics with predicted acceleration based on state """
    def __init__(self, data_blob, accel_blob, output_blob, djnt, dee, du):
        super(AccelLayer, self).__init__([data_blob, accel_blob], output_blob)  
        self.data_blob = data_blob
        self.accel_blob = accel_blob

        idx = 0
        self.djnt = djnt
        self.dee = dee
        self.du = du
        self.dx = 2*djnt+2*dee
        self.idxpos = slice(idx,idx+djnt); idx+=djnt
        self.idxvel = slice(idx,idx+djnt); idx+=djnt
        self.idxeepos = slice(idx,idx+dee); idx+=dee
        self.idxeevel = slice(idx,idx+dee); idx+=dee
        self.idxu = slice(self.dx,self.dx+du)

        self.construct_forward_matrices()

    def construct_forward_matrices(self):
        t = 0.05
        forward_mat = np.zeros((self.dx+self.du, self.dx))
        forward_mat[self.idxpos, self.idxpos] = np.eye(self.djnt)
        forward_mat[self.idxvel, self.idxvel] = np.eye(self.djnt)
        forward_mat[self.idxvel, self.idxpos] = t*np.eye(self.djnt)
        forward_mat[self.idxeepos, self.idxeepos] = np.eye(self.dee)
        forward_mat[self.idxeevel, self.idxeevel] = np.eye(self.dee)
        forward_mat[self.idxeevel, self.idxeepos] = t*np.eye(self.dee)
        self.forward_mat = theano.shared(forward_mat.astype(np.float32), name="AccLayer_forward_mat_"+str(self.layer_id))

        jnt_mat = np.zeros((self.djnt+self.dee, self.dx))
        jnt_mat[:self.djnt, self.idxpos] = t*t*np.eye(self.djnt)
        jnt_mat[:self.djnt, self.idxvel] = t*np.eye(self.djnt)
        jnt_mat[self.djnt:self.djnt+self.dee, self.idxeepos] = t*t*np.eye(self.dee)
        jnt_mat[self.djnt:self.djnt+self.dee, self.idxeevel] = t*np.eye(self.dee)
        self.jnt_mat = theano.shared(jnt_mat.astype(np.float32), name="AccLayer_acc_mat_"+str(self.layer_id))

    def forward(self, input_data):
        acc = input_data[self.accel_blob]
        jnts_data = input_data[self.data_blob]
        return jnts_data.dot(self.forward_mat) + acc.dot(self.jnt_mat)

    def params(self):
        return []

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

def train_gd_rmsprop(obj, params, args, updates=None, eps=1e-6, weight_decay=0.0):
    gradients = T.grad(obj, params)
    eta = T.scalar('lr')
    rho = T.scalar('rho')
    momentum = T.scalar('momentum')
    accs = [theano.shared(np.copy(param.get_value())*0.0) for param in params]
    momentums = [theano.shared(np.copy(param.get_value())*0.0) for param in params]
    #if updates is None:
    #TODO: Incorporate updates. For some reason the state magically updates
    updates = []
    for i in range(len(gradients)):
        acc_new = rho*accs[i] + (1-rho) * gradients[i] ** 2
        gradient_scale = T.sqrt(acc_new + eps)
        new_grad = (gradients[i] + weight_decay*params[i])/gradient_scale

        updated_gradient = (new_grad)+momentum*momentums[i]

        updates.append((accs[i], gpu_host(acc_new)))
        updates.append((params[i], gpu_host(params[i]-(eta)*updated_gradient)))
        updates.append((momentums[i], gpu_host(updated_gradient)))

    train = theano.function(
        inputs=args+[eta, rho, momentum],
        outputs=[obj],
        updates=updates,
        on_unused_input='warn'
    )
    return train

def train_gd_momentum(obj, params, args, updates=None, scl=1.0, weight_decay=0.0):
    obj = obj
    scl = float(scl)
    gradients = T.grad(obj, params)
    eta = T.scalar('lr')
    momentum = T.scalar('momentum')
    momentums = [theano.shared(np.copy(param.get_value())*0.0) for param in params]
    #if updates is None:
    #TODO: Incorporate updates. For some reason the state magically updates
    updates = []
    for i in range(len(gradients)):
        update_gradient = (gradients[i])+momentum*momentums[i]+weight_decay*params[i]
        updates.append((params[i], gpu_host(params[i]-(eta/scl)*update_gradient)))
        updates.append((momentums[i], gpu_host(update_gradient)))

    train = theano.function(
        inputs=args+[eta, momentum],
        outputs=[obj],
        updates=updates,
        on_unused_input='warn'
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
        self.batch = batch
        for input_var in self.inputs:
            batch.set_data(input_var.name, input_var)

        updates = []
        for layer in self.layers:
            updates.extend(list(layer.forward_batch(batch)))
        obj = self.loss.forward_batch(batch)
        return obj, updates

    def set_net_inputs(self, inputs):
        self.inputs = inputs

    def get_train_function(self, objective, updates=None, type='sgd', weight_decay=0):
        if type=='sgd':
            return train_gd_momentum(objective, self.params, self.inputs, weight_decay=weight_decay, updates=updates)
        elif type=='rmsprop':
            return train_gd_rmsprop(objective, self.params, self.inputs, weight_decay=weight_decay, updates=updates)
        else:
            raise NotImplementedError()

    def get_loss_function(self, objective, updates=None):
        return theano.function(inputs=self.inputs, outputs=[objective], updates=updates, on_unused_input='warn')

    def update(self, stage=STAGE_TRAIN):
        for layer in self.layers:
            layer.update(stage=stage)

    def get_output(self, var_name, inputs=None, updates=None):
        if inputs is None:
            fn_inputs = self.inputs
        else:
            fn_inputs = [self.batch.get_data(input_name) for input_name in inputs]
        return theano.function(inputs=fn_inputs, outputs=[self.batch.get_data(var_name)], updates=updates, on_unused_input='warn')

    def get_jac(self, output_name, input_name, inputs=None, updates=None):
        """ Return derivative of one variable (output_name) w.r.t. another (input_name) """
        jacobian = theano.gradient.jacobian(self.batch.get_data(output_name)[0], self.batch.get_data(input_name))
        if inputs is None:
            jac_inputs = self.inputs
        else:
            jac_inputs = [self.batch.get_data(input_name) for input_name in inputs]
        jac_fn = theano.function(inputs=jac_inputs, outputs=jacobian, on_unused_input='warn')
        return jac_fn

    def get_recurrent_state(self):
        state = [None]*len(self.layers)
        for i in range(len(self.layers)):
            state[i] = self.layers[i].get_state_value()
        return state

    def clear_recurrent_state(self):
        state_list = self.get_recurrent_state()
        for i in range(len(state_list)):
            if state_list[i] is not None:
                state_list[i].fill(0.0)
        self.set_recurrent_state(state_list)

    def set_recurrent_state(self, state):
        assert(len(state) == len(self.layers))
        for i in range(len(self.layers)):
            self.layers[i].set_state_value(state[i])

    def __getstate__(self):  # For pickling
        return (self.layers, self.loss)

    def __setstate__(self, state):  # For pickling
        self.__init__(state[0], state[1])

    def pickle(self, fname):
        with open(fname, 'w') as pklfile:
            cPickle.dump(self, pklfile)
        LOGGER.debug('Dumped network to: %s', fname)

class RecurrentDynamicsNetwork(Network):
    def __init__(self, layers, loss):
        super(RecurrentDynamicsNetwork, self).__init__(layers, loss)

    def init_functions(self, output_blob='rnn_out', train_algo='rmsprop', weight_decay=1e-5):
        self.set_net_inputs([T.matrix('data'), T.matrix('lbl'), T.vector('clip')])
        obj, updates = self.symbolic_forward()
        self.train_gd = self.get_train_function(obj, updates, type=train_algo, weight_decay=weight_decay)
        self.total_obj = self.get_loss_function(obj)

        self.rnn_out = self.get_output(output_blob, inputs=['data', 'clip'], updates=updates)
        jac = self.get_jac(output_blob, 'data', inputs=['data', 'clip'])
        def taylor_expand(data_pnt, clip=1):
            clip = clip*np.ones(1).astype(np.float32)
            data_pnt_exp = np.expand_dims(data_pnt, axis=0).astype(np.float32)
            F = jac(data_pnt_exp, clip)[:,0,:]
            net_fwd = self.rnn_out(data_pnt_exp, clip)[0][0]
            f = -F.dot(data_pnt) + net_fwd
            return F, f
        self.getF = taylor_expand

    def fwd_single(self, xu, clip=1):
        clip = clip*np.ones(1).astype(np.float32)
        data_pnt_exp = np.expand_dims(xu, axis=0).astype(np.float32)
        net_fwd = self.rnn_out(data_pnt_exp, clip)[0][0]
        return net_fwd

    def loss_single(self, xu, xnext):
        clip = np.zeros(1).astype(np.float32)
        data_pnt_exp = np.expand_dims(xu, axis=0).astype(np.float32)
        label_exp = np.expand_dims(xnext, axis=0).astype(np.float32)
        obj = self.total_obj(data_pnt_exp, label_exp, clip)[0]
        return obj

    def fwd_batch(self, xu, clip):
        net_fwd = self.rnn_out(xu, clip)
        return net_fwd

    def to_feedforward_test(self):
        LOGGER.debug("Converting recurrent network to feedforward...")       
        layers = []
        for layer in self.layers:
            if hasattr(layer, 'to_feedforward_test'):
                new_layer = layer.to_feedforward_test()
                LOGGER.debug("Transformed layer %s", layer)
            else:
                new_layer = layer
                LOGGER.debug("Copied layer %s", layer)
            layers.append(new_layer)
        return RecurrentTestNetwork(layers, self.loss)
     
class RecurrentTestNetwork(Network):
    """ Hacky way to convert a recurrent net into a feedforward net during online execution"""
    def __init__(self, layers, loss):
        super(RecurrentTestNetwork, self).__init__(layers, loss)
        self.hidden_state_plot = None

    def init_functions(self, output_blob='rnn_out'):
        hidden_state_blobs = []
        hidden_state_vars = []
        hidden_state_outputs = []
        for layer in self.layers:
            if isinstance(layer, FF_RNNHackLayer):
                hidden_in, hidden_out = layer.hidden_state_io_blobs()
                hidden_state_blobs.append(hidden_in)
                hidden_state_vars.append(T.vector(hidden_in))
                hidden_state_outputs.append(hidden_out)

        self.set_net_inputs([T.matrix('data'), T.matrix('lbl')] + hidden_state_vars)
        obj, updates = self.symbolic_forward()
        #self.train_gd = self.get_train_function(obj, updates)
        self.total_obj = self.get_loss_function(obj)

        self.__rnn_out_fn = self.get_output_and_state([output_blob]+hidden_state_outputs, inputs=['data']+hidden_state_blobs, updates=updates)
        self.__jac = self.get_jac(output_blob, 'data', inputs=['data']+hidden_state_blobs)

    def getF(self, data_pnt, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.get_init_hidden_state()
        data_pnt_exp = np.expand_dims(data_pnt, axis=0).astype(np.float32)
        F = self.__jac(*([data_pnt_exp]+hidden_state))[:,0,:]
        net_outs = self.__rnn_out_fn(*([data_pnt_exp]+hidden_state))
        net_fwd = net_outs[0][0]
        hidden_state = net_outs[1:]
        for i in range(len(hidden_state)):
            hidden_state[i] = hidden_state[i][0]
        f = -F.dot(data_pnt) + net_fwd
        return F, f, hidden_state

    def get_output_and_state(self, out_vars, inputs=None, updates=None):
        if inputs is None:
            fn_inputs = self.inputs
        else:
            fn_inputs = [self.batch.get_data(input_name) for input_name in inputs]
        outputs = []
        for out_var in out_vars:
            outputs.append(self.batch.get_data(out_var))
        return theano.function(inputs=fn_inputs, outputs=outputs, updates=updates, on_unused_input='warn')

    def get_init_hidden_state(self):
        init_state = []
        for layer in self.layers:
            if isinstance(layer, FF_RNNHackLayer):
                init_state.append(layer.init_recurrent_state().astype(np.float32))
        return init_state

    def fwd_single(self, xu, hidden_state):
        data_pnt_exp = np.expand_dims(xu, axis=0).astype(np.float32)
        net_outs = self.__rnn_out_fn(*([data_pnt_exp]+hidden_state))
        net_fwd = net_outs[0][0]
        hidden_state = net_outs[1:]
        for i in range(len(hidden_state)):
            hidden_state[i] = hidden_state[i][0]
        return net_fwd, hidden_state

    def loss_single(self, xu, xnext, hidden_state):
        clip = np.zeros(1).astype(np.float32)
        data_pnt_exp = np.expand_dims(xu, axis=0).astype(np.float32)
        label_exp = np.expand_dims(xnext, axis=0).astype(np.float32)
        obj = self.total_obj(*([data_pnt_exp, label_exp]+hidden_state))[0]
        return obj, hidden_state


def unpickle_net(fname):
    with open(fname, 'r') as pklfile:
        net = cPickle.load(pklfile)
    LOGGER.debug('Loaded network from: %s', fname)
    return net

def rnntest():
    np.random.seed(123)
    logging.basicConfig(level=logging.DEBUG)

    bsize = 20
    N = 200

    data = np.zeros((N, 10))
    label = np.zeros((N, 10))
    clip = np.ones((N,)).astype(np.float32)
    tmp = None
    for i in range(N):
        if i%5 == 0:
            tmp = np.random.randn(10)
            data[i] = tmp
            clip[i] = 0
        label[i] = tmp
    data = data.astype(np.float32)
    label = label.astype(np.float32)

    ip1 = RNNIPLayer('data', 'ip1', 'clip', 10, 10) 
    loss = SquaredLoss('ip1', 'lbl')
    net = RecurrentDynamicsNetwork([ip1], loss)
    net.init_functions(output_blob='ip1')
    #net = unpickle_net('test.pkl')

    #net.set_net_inputs([T.matrix('data'), T.matrix('lbl'), T.vector('clip')])
    #obj, updates = net.symbolic_forward()
    #train_gd = net.get_train_function(obj, updates)
    #total_obj = net.get_loss_function(obj, updates)

    lr = 4e-3/bsize
    lr_schedule = {
        400000: 0.2,
        800000: 0.2,
    }
    epochs = 0
    for i in range(20000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            #perm = np.random.permutation(N)
            #data = data[perm]
            #label = label[perm]
            #clip = clip[perm]
            continue
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _clip = clip[bstart:bend]
        objval = net.train_gd(_data, _label, _clip, lr, 0.90)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 500 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            #ip1= rnn_out(_data, _clip)
        if i % 10000 == 0:
            pass
            #if i>0:
            #    net.pickle('test.pkl')
            #total_err = total_obj(data, label, clip)
            #print 'Total train error:', total_err

    test_data = np.zeros((5, 10)).astype(np.float32)
    test_lbl = np.zeros((5, 10)).astype(np.float32)
    test_state = np.random.randn(10).astype(np.float32)

    net.layers[0].set_state_value(np.ones(10).astype(np.float32))
    e1 = net.eval_forward(np.zeros(10))
    e2 = net.eval_forward(np.zeros(10))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    rnntest()
