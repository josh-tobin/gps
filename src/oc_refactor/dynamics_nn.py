import numpy as np
import theano
import cPickle
import logging
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

NN_INIT_WT = 0.08
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
    'relu': lambda x: x*(x>0),
    'resqrt': lambda x: (T.sqrt(x)-1.0)*(x>1.0),
    'relog': lambda x: (T.log(x))*(x>1.0),
    'tanh': T.tanh,
    'none': lambda x: x
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
        return T.tanh(prev_layer)


"""
class GRULayer(RecurrentLayer):
    def __init__(self, input_blob, output_blob, clip_blob, din, dout, activation=None):
        super(GRULayer, self).__init__(input_blob, output_blob, clip_blob)    
        self.input_blob = input_blob
        self.output_blob = output_blob
        self.din = din
        self.dout = dout
        if activation is None:
            self.activation='tanh'
        else:
            self.activation=activation
        self.wff = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wff_'+str(self.layer_id))
        self.wr = theano.shared(NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wr_'+str(self.layer_id))
        self.wgate_in = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wgatein_'+str(self.layer_id))
        self.wgate_hidden = theano.shared(NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wgatehidden_'+str(self.layer_id))
        self.wreset_in = theano.shared(NN_INIT_WT*np.random.randn(din, dout).astype(np.float32), name='rnn_g_wresetin_'+str(self.layer_id))
        self.wreset_hidden = theano.shared(NN_INIT_WT*np.random.randn(dout, dout).astype(np.float32), name='rnn_g_wresethidden_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='rnn_g_b_'+str(self.layer_id))
        self.state = theano.shared(np.zeros(dout).astype(np.float32), name='rnn_g_state_'+str(self.layer_id))

    def forward(self, prev_layer, prev_state):
        gate_value = T.nnet.sigmoid(self.wgate_in.T.dot(prev_layer)+self.wgate_hidden.T.dot(prev_state))
        reset_value = T.nnet.sigmoid(self.wreset_in.T.dot(prev_layer)+self.wreset_hidden.T.dot(prev_state))
        new_state = ACTIVATION_DICT[self.activation](self.wff.T.dot(prev_layer) + self.wr.T.dot(reset_value*prev_state) + self.b)
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
        new_layer = FF_GRULayer(self.input_blob, self.output_blob, self.din, self.dout, activation=self.activation)
        new_layer.wff.set_value(self.wff.get_value())
        new_layer.wr.set_value(self.wr.get_value())
        new_layer.wgate_in.set_value(self.wgate_in.get_value())
        new_layer.wgate_hidden.set_value(self.wgate_hidden.get_value())

        #AAA
        new_layer.wreset_in.set_value(self.wreset_in.get_value())
        new_layer.wreset_hidden.set_value(self.wreset_hidden.get_value())

        new_layer.b.set_value(self.b.get_value())
        return new_layer
"""

class FFIPLayer(FeedforwardLayer):
    def __init__(self, input_blob, output_blob, din, dout):
        super(FFIPLayer, self).__init__(input_blob, output_blob)  
        self.input_blob = input_blob
        w_init = np.sqrt(2.0/din)*np.random.randn(din, dout).astype(np.float32)
        self.w = theano.shared(w_init, name='ff_ip_w_'+str(self.layer_id))
        self.b = theano.shared(0*np.random.randn(dout).astype(np.float32), name='ff_ip_b_'+str(self.layer_id))

    def forward(self, input_data, stage=STAGE_TRAIN):
        prev_layer = input_data[self.input_blob]
        return prev_layer.dot(self.w)+self.b

    def params(self):
        return [self.w, self.b]

class PrevSALayer(FeedforwardLayer):
    def __init__(self, input_blob, prev_sa_blob, output_blob, din, dout, dpxu):
        super(PrevSALayer, self).__init__([input_blob, prev_sa_blob], output_blob)  
        self.input_blob = input_blob
        self.prev_sa_blob = prev_sa_blob
        w_init = np.sqrt(2.0/din)*np.random.randn(din, dout).astype(np.float32)
        self.w = theano.shared(w_init, name='psa_ip_w_'+str(self.layer_id))
        w_init = np.sqrt(2.0/(dpxu))*np.random.randn(dpxu, dout).astype(np.float32)
        self.wsa = theano.shared(w_init, name='psa_ip_wsa_'+str(self.layer_id))
        self.b = theano.shared(0*np.ones(dout).astype(np.float32), name='psa_ip_b_'+str(self.layer_id))

    def forward(self, input_data, stage=STAGE_TRAIN):
        prev_layer = input_data[self.input_blob]
        prev_state_action = input_data[self.prev_sa_blob]
        return prev_layer.dot(self.w)+prev_state_action.dot(self.wsa)+self.b

    def params(self):
        return [self.w, self.wsa, self.b]

class PrevSALayer2(FeedforwardLayer):
    def __init__(self, input_blob, prev_sa_blob, output_blob, dx, du, dout):
        super(PrevSALayer2, self).__init__([input_blob, prev_sa_blob], output_blob)  
        self.input_blob = input_blob
        self.prev_sa_blob = prev_sa_blob
        self.dx = dx
        w_init = np.sqrt(2.0/du)*np.random.randn(du, dout).astype(np.float32)
        self.wact = theano.shared(w_init, name='psa_ip_wact_'+str(self.layer_id))
        w_init = np.sqrt(2.0/(dx))*np.random.randn(dx, dout).astype(np.float32)
        self.wdiff = theano.shared(w_init, name='psa_ip_wdiff_'+str(self.layer_id))
        self.b = theano.shared(0*np.ones(dout).astype(np.float32), name='psa_ip_b_'+str(self.layer_id))

    def forward(self, input_data, stage=STAGE_TRAIN):
        prev_layer = input_data[self.input_blob]
        prev_state_action = input_data[self.prev_sa_blob]
        diff = prev_layer[:,:self.dx] - prev_state_action
        action = prev_layer[:,self.dx:]
        return action.dot(self.wact)+diff.dot(self.wdiff)+self.b

    def params(self):
        return [self.wdiff, self.wact, self.b]

class SubtractAction(FeedforwardLayer):
    def __init__(self, input_blob, prev_sa_blob, output_blob):
        super(SubtractAction, self).__init__([input_blob, prev_sa_blob], output_blob)  
        self.input_blob = input_blob
        self.prev_sa_blob = prev_sa_blob

    def forward(self, input_data, stage=STAGE_TRAIN):
        prev_layer = input_data[self.input_blob]
        prev_state_action = input_data[self.prev_sa_blob]

        new_sa = prev_layer[:,-7:] - prev_state_action[:,-7:]
        new_sa = T.concatenate((prev_layer[:,:-7], new_sa), axis=1)
        return new_sa

    def params(self):
        return []

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

class Loss(object):
    def __init__(self):
        pass
    def forward_batch(self, batch):
        raise NotImplementedError()


class SquaredLoss(Loss):
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
        return obj, []


def train_gd_rmsprop(obj, params, args, extra_outputs=None, updates=None, eps=1e-6, weight_decay=0.0):
    if extra_outputs is None:
        extra_outputs=[]
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
        outputs=[obj]+extra_outputs,
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

        if isinstance(self.loss, Loss):
            obj, new_updates = self.loss.forward_batch(batch)
            updates.extend(new_updates)
        else: #Assume list
            obj, new_updates = self.loss[0].forward_batch(batch)
            updates.extend(new_updates)
            for i in range(1,len(self.loss)):
                tmp_obj, new_updates = self.loss[i].forward_batch(batch)
                updates.extend(new_updates)
                obj += tmp_obj
        return obj, updates

    def varname_to_symbolic(self, varname):
        return [self.batch.get_data(var) for var in varname]

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

    def __getstate__(self):  # For pickling
        return (self.layers, self.loss)

    def __setstate__(self, state):  # For pickling
        self.__init__(state[0], state[1])

    def pickle(self, fname):
        with open(fname, 'w') as pklfile:
            cPickle.dump(self, pklfile)
        LOGGER.debug('Dumped network to: %s', fname)

class PrevSADynamicsNetwork(Network):
    def __init__(self, layers, loss, sigmax=None):
        super(PrevSADynamicsNetwork, self).__init__(layers, loss)
        self.sigmax = sigmax

    def init_functions(self, output_blob='rnn_out', train_algo='rmsprop', weight_decay=1e-5):
        self.set_net_inputs([T.matrix('data'), T.matrix('prevxu'), T.matrix('lbl')])
        obj, updates = self.symbolic_forward()
        self.train_gd = self.get_train_function(obj, updates, type=train_algo, weight_decay=weight_decay)
        self.total_obj = self.get_loss_function(obj)

        self.ff_out = self.get_output(output_blob, inputs=['data', 'prevxu'], updates=updates)
        jac = self.get_jac(output_blob, 'data', inputs=['data', 'prevxu'])
        def taylor_expand(data_pnt, prevxu):
            pxu_exp = np.expand_dims(prevxu, axis=0).astype(np.float32)
            data_pnt_exp = np.expand_dims(data_pnt, axis=0).astype(np.float32)    
            F = jac(data_pnt_exp, pxu_exp)[:,0,:]
            net_fwd = self.ff_out(data_pnt_exp, pxu_exp)[0][0]
            f = -F.dot(data_pnt) + net_fwd
            return F, f
        self.linearize = taylor_expand

    def fwd_single(self, xu, prevxu):
        pxu_exp = np.expand_dims(prevxu, axis=0).astype(np.float32)
        data_pnt_exp = np.expand_dims(xu, axis=0).astype(np.float32)
        net_fwd = self.ff_out(data_pnt_exp, pxu_exp)[0]
        return net_fwd

    def loss_single(self, xu, prevxu, tgt):
        """Run SGD on a single (vector) data point """
        pxu_exp = np.expand_dims(prevxu, axis=0).astype(np.float32)
        data_pnt_exp = np.expand_dims(xu, axis=0).astype(np.float32)
        tgt = np.expand_dims(tgt, axis=0).astype(np.float32)
        return self.total_obj(xu, prevxu, tgt)

    def fwd_batch(self, xu, pxu):
        net_fwd = self.ff_out(xu, pxu)
        return net_fwd

    def calculate_sigmax(self, xu, prevxu, tgt):
        predictions = self.ff_out(xu, prevxu)[0]
        diff = predictions-tgt
        self.sigmax = np.cov(diff.T)
        #self.sigmax = predictions.T.dot(predictions) - tgt.T.dot(tgt)

    def __getstate__(self):  # For pickling
        return (self.layers, self.loss, self.sigmax)

    def __setstate__(self, state):  # For pickling
        self.__init__(state[0], state[1], sigmax=state[2])



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
