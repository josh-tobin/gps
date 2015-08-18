import time
import numpy as np
import sys
import cPickle
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.cuda.basic_ops as tcuda
import scipy.io
ONE = np.array([1])

NN_INIT = 0.1

gpu_host = tcuda.gpu_from_host


def dump_net(fname, net):
    with open(fname, 'w') as f:
        cPickle.dump(net.serialize(), f)  

def load_net(fname):
    net = NNetDyn([], None, init_funcs=False)
    with open(fname, 'r') as f:
        wts = cPickle.load(f)
        net.unserialize(wts)
        net.init_funcs()
    return net

class NNetDyn(object):
    def __init__(self, layers, loss_wt, post_layer=None, init_funcs=True,weight_decay=0.0, layer_reg = []):
        self.params = []
        self.loss_wt = loss_wt
        self.layer_reg = layer_reg
        self.nparams = 0
        self.layers = layers
        self.post_layer = post_layer
        self.weight_decay = weight_decay
        if init_funcs:
            self.init_funcs()

    def init_funcs(self):
        self.params = []
        self.nparams = 0
        for layer in self.layers:
            self.params.extend(layer.params())
            self.nparams += layer.n_params()

        #Set up matrix functions
        data = T.matrix('mdata')
        lbl = T.matrix('mlbl')
        net_out, layers_out = self.fwd(data, training=True)

        extra_loss = []
        loss_wts = []
        for lreg in self.layer_reg:
            layer_idx = lreg['layer_idx']
            wt = lreg['l2wt']
            print 'Adding layer regularization of %f for layer %d!' % (wt, layer_idx)
            layer = layers_out[layer_idx]
            extra_loss.append(L2Reg(layer))
            loss_wts.append(wt)

        self.loss = SquaredLoss(self.loss_wt)
        if extra_loss:
            extra_loss.append(self.loss)
            loss_wts.append(1.0)
            self.loss = SumLoss(extra_loss, loss_wts)
        obj = self.loss.loss(lbl, net_out)
        self.obj = theano.function(inputs=[data, lbl], outputs=obj, on_unused_input='warn')
        self.train_sgd = train_gd_momentum(obj, self.params, [data, lbl], scl=self.nparams, weight_decay=self.weight_decay)


        # Set up vector functions
        data = T.vector('vdata')
        lbl = T.vector('vlbl')
        net_out, _ = self.fwd(data, training=False)

        self.net_out_vec = theano.function(inputs=[data], outputs=net_out)
        self.out_shape = theano.function(inputs=[data], outputs=net_out.shape)
        data_grad = theano.gradient.jacobian(net_out, data)
        self.jac = theano.function(inputs=[data], outputs=data_grad)

    def serialize(self):
        wts = []
        for layer in self.layers:
            wts.append(layer)
        metadata = {
            'time': time.time(),
            'loss_wt': self.loss_wt,
            'weight_decay': self.weight_decay,
            'post_layer': self.post_layer,
            'layer_reg': self.layer_reg
        }
        return (metadata, wts)

    def unserialize(self, net):
        metadata = net[0]
        wts = net[1]
        #self.loss = metadata['loss']
        self.loss_wt = metadata['loss_wt']
        print 'Loss wt:', self.loss_wt
        self.layer_reg = metadata.get('layer_reg', [])

        self.post_layer = metadata.get('post_layer', None)

        self.weight_decay = metadata['weight_decay']
        print 'Weight Decay:', self.weight_decay
        self.layers = [None]*len(wts)
        for i in range(len(wts)):
            self.layers[i] = wts[i]

    def fwd(self, data, training=True):
        """Generate symbolic expressions for forward pass"""
        net_out = data
        layers_out = []
        for layer in self.layers:
            layer.set_input_data(data)
            net_out = layer.forward(net_out, training=training)
            layers_out.append(net_out)
        layers_out.append(net_out)
        #if post_layer and self.post_layer:
        #    net_out = self.post_layer.forward(net_out, training=training)
        return net_out , layers_out

    def fwd_single(self, xu, layer=None, training=False):
        """ Run forward pass on a single data point and return numeric output """
        if layer is not None:
            data = T.vector('data')
            _, layers_out = self.fwd(data, training=training)
            layer_out = layers_out[layer]
            net_out_vec = theano.function(inputs=[data], outputs=layer_out)
            return net_out_vec(xu)
        #xu = np.concatenate((xu, ONE))
        return self.net_out_vec(xu)

    def train_single(self, xu, tgt, lr, momentum):
        """Run SGD on a single (vector) data point """
        xu = np.expand_dims(xu, axis=0).astype(np.float32)
        tgt = np.expand_dims(tgt, axis=0).astype(np.float32)
        return self.train_sgd(xu, tgt, lr, momentum)[0]

    def train(self, xu, tgt, lr, momentum):
        """ Run SGD on matrix-formatted data (NxD) """
        #xu = np.c_[xu, np.ones((xu.shape[0],1))]
        return self.train_sgd(xu, tgt, lr, momentum)

    def obj_matrix(self, xu, tgt):
        """   """
        #xu = np.c_[xu, np.ones((xu.shape[0],1))]
        return self.obj(xu, tgt)

    def obj_vec(self, xu, tgt):
        """   """
        xu = np.expand_dims(xu, axis=0).astype(np.float32)
        tgt = np.expand_dims(tgt, axis=0).astype(np.float32)
        return self.obj_matrix(xu, tgt)

    def getF(self, xu):
        """ Return F, f - 1st order Taylor expansion of network around xu """
        F = self.jac(xu)
        f = -F.dot(xu)+self.fwd_single(xu)
        return F, f

    def __str__(self):
        return str(self.layers)


def train_gd(obj, params, args):
    obj = obj
    eta = T.scalar('lr')
    params = params
    gradients = T.grad(obj, params)
    updates = [None]*len(gradients)
    for i in range(len(gradients)):
        updates[i] = (params[i], params[i]-eta*gradients[i])
    train = theano.function(
        inputs=args+[eta],
        outputs=[obj],
        updates=updates
    )
    return train    

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

class Layer(object):
    def __init__(self):
        pass

    def set_input_data(self, data_in):
        self.input_data = data_in

    def params(self):
        """ Return a list of trainable parameters """
        return []
    
    def n_params(self):
        return 0

    def serialize(self):
        return []

    def unserialize(self, wts):
        pass

class NormalizeLayer(Layer):
    def __init__(self):
        self.reverse = False
        self.mean = 0
        self.sig = 0

    def forward(self, prev_layer, training=True):
        if self.reverse:
            return prev_layer*self.sig + self.mean
        else:
            return (prev_layer-self.mean)/self.sig

    def generate_weights(self, data, reverse=False):
        self.mean = np.mean(data, axis=0)
        data = data-self.mean
        sig = np.std(data, axis=0)
        self.sig = sig
        self.reverse = reverse

    def serialize(self):
        return [self.sig, self.mean, self.reverse]

    def unserialize(self, wts):
        self.sig = wts[0]
        self.mean = wts[1]
        self.reverse = self.reverse

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())

class WhitenLayer(Layer):
    def __init__(self):
        self.reverse = False
        self.mean = 0
        self.w = 0

    def forward(self, prev_layer, training=True):
        if self.reverse:
            return (prev_layer.dot(self.w))+self.mean
        else:
            return (prev_layer-self.mean).dot(self.w)

    def generate_weights(self, data, reverse=False):
        self.mean = np.mean(data, axis=0).astype(np.float32)
        data = data-self.mean
        covar = np.cov(data.T)
        U,S,V = np.linalg.svd(covar)
        self.w = np.diag(1.0/(np.sqrt(S + 1e-4))).dot(U.T).T.astype(np.float32)
        if self.reverse:
            self.w = np.linalg.inv(self.w)
        self.reverse = reverse

    def serialize(self):
        return [self.sig, self.mean, self.reverse]

    def unserialize(self, wts):
        self.sig = wts[0]
        self.mean = wts[1]
        self.reverse = self.reverse

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())

class FFIPLayer(Layer):
    """ Feedforward inner product layer """
    n_instances = 0
    def __init__(self, n_in, n_out):
        FFIPLayer.n_instances += 1
        self.n_in = n_in
        self.n_out = n_out
        self.layer_id = FFIPLayer.n_instances
        self.w = theano.shared(NN_INIT*np.random.randn(n_in, n_out).astype(np.float32), name="ff_ip_w_"+str(self.layer_id))
        self.b = theano.shared(NN_INIT*np.random.randn(n_out).astype(np.float32), name="b_ip"+str(self.layer_id))

    def forward(self, prev_layer, training=True):
        return prev_layer.dot(self.w) + self.b

    def params(self):
        """ Return a list of trainable parameters """
        return [self.w, self.b]
    
    def n_params(self):
        return self.n_in*self.n_out+self.n_out

    def serialize(self):
        return [self.w.get_value(), self.b.get_value()]

    def unserialize(self, wts):
        self.w.set_value(wts[0])
        self.b.set_value(wts[1])

    def set_weights(self, wt, bias):
        self.w.set_value(wt)
        self.b.set_value(bias)

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())
        #return "W:%s" % (self.w.get_value())

class AccelLayer(Layer):
    """ Acceleration Layer """
    def __init__(self):
        self.t = 0.05
        self.idxpos = slice(0,7)
        self.idxvel = slice(7,14)
        self.idxprevu = slice(14,21)
        self.idxeepos = slice(21,30)
        self.idxeevel = slice(30,39)
        self.idxu = slice(39,46)

    def forward(self, prev_layer, training=True):
        if training:
            jnt_accel = prev_layer[:,:7]
            ee_accel = prev_layer[:,7:16]
            t = self.t
            jnt_pos = self.input_data[:,self.idxpos] + t*self.input_data[:,self.idxvel] + 0.5*t*t*jnt_accel
            jnt_vel = self.input_data[:,self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[:,self.idxeepos]+ t*self.input_data[:,self.idxeevel] + 0.5*t*t*ee_accel
            ee_vel = self.input_data[:,self.idxeevel] + t*ee_accel
            prev_action = self.input_data[:,self.idxu]
            return T.concatenate([jnt_pos, jnt_vel, prev_action, ee_pos, ee_vel], axis=1)
        else:
            jnt_accel = prev_layer[:7]
            ee_accel = prev_layer[7:16]
            t = self.t
            jnt_pos = self.input_data[self.idxpos] + t*self.input_data[self.idxvel] + 0.5*t*t*jnt_accel
            jnt_vel = self.input_data[self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[self.idxeepos]+ t*self.input_data[self.idxeevel] + 0.5*t*t*ee_accel
            ee_vel = self.input_data[self.idxeevel] + t*ee_accel
            prev_action = self.input_data[self.idxu]
            return T.concatenate([jnt_pos, jnt_vel, prev_action, ee_pos, ee_vel])

    def __str__(self):
        return "AccelLayer"

class AccelLayerMJC(Layer):
    """ Acceleration Layer """
    def __init__(self):
        self.t = 0.05
        self.idxpos = slice(0,7)
        self.idxvel = slice(7,14)
        self.idxeepos = slice(14,20)
        self.idxeevel = slice(20,26)
        self.idxu = slice(26,33)

    def forward(self, prev_layer, training=True):
        if training:
            jnt_accel = prev_layer[:,:7]
            ee_accel = prev_layer[:,7:13]
            t = self.t
            jnt_pos = self.input_data[:,self.idxpos] + t*self.input_data[:,self.idxvel] + 0.5*t*t*jnt_accel
            jnt_vel = self.input_data[:,self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[:,self.idxeepos]+ t*self.input_data[:,self.idxeevel] + 0.5*t*t*ee_accel
            ee_vel = self.input_data[:,self.idxeevel] + t*ee_accel
            return T.concatenate([jnt_pos, jnt_vel, ee_pos, ee_vel], axis=1)
        else:
            jnt_accel = prev_layer[:7]
            ee_accel = prev_layer[7:13]
            t = self.t
            jnt_pos = self.input_data[self.idxpos] + t*self.input_data[self.idxvel] + 0.5*t*t*jnt_accel
            jnt_vel = self.input_data[self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[self.idxeepos]+ t*self.input_data[self.idxeevel] + 0.5*t*t*ee_accel
            ee_vel = self.input_data[self.idxeevel] + t*ee_accel
            return T.concatenate([jnt_pos, jnt_vel, ee_pos, ee_vel])

    def __str__(self):
        return "AccelLayer"

class AccelLayerFT(Layer):
    """ Acceleration Layer """
    def __init__(self):
        self.t = 0.05
        self.idxpos = slice(0,7)
        self.idxvel = slice(7,14)
        self.idxprevu = slice(14,21)
        self.idxft = slice(21,27)
        self.idxeepos = slice(27,36)
        self.idxeevel = slice(36,45)
        self.idxu = slice(45,52)

    def forward(self, prev_layer, training=True):
        if training:
            jnt_accel = prev_layer[:,:7]
            ee_accel = prev_layer[:,7:16]
            nextft = prev_layer[:,16:22]
            t = self.t
            jnt_pos = self.input_data[:,self.idxpos] + t*self.input_data[:,self.idxvel] + 0.5*t*t*jnt_accel
            jnt_vel = self.input_data[:,self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[:,self.idxeepos]+ t*self.input_data[:,self.idxeevel] + 0.5*t*t*ee_accel
            ee_vel = self.input_data[:,self.idxeevel] + t*ee_accel
            prev_action = self.input_data[:,self.idxu]
            return T.concatenate([jnt_pos, jnt_vel, prev_action, nextft, ee_pos, ee_vel], axis=1)
        else:
            jnt_accel = prev_layer[:7]
            ee_accel = prev_layer[7:16]
            nextft = prev_layer[16:22]
            t = self.t
            nextft = self.input_data[self.idxft]+t*nextft
            jnt_pos = self.input_data[self.idxpos] + t*self.input_data[self.idxvel] + 0.5*t*t*jnt_accel
            jnt_vel = self.input_data[self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[self.idxeepos]+ t*self.input_data[self.idxeevel] + 0.5*t*t*ee_accel
            ee_vel = self.input_data[self.idxeevel] + t*ee_accel
            prev_action = self.input_data[self.idxu]
            return T.concatenate([jnt_pos, jnt_vel, prev_action, nextft, ee_pos, ee_vel])

    def __str__(self):
        return "AccelLayer"

class DropoutLayer(Layer):
    def __init__(self, n_in, rngseed=10, p=0.5):
        self.n_in = n_in
        self.rng = RandomStreams(seed=rngseed)
        self.p = p

    def forward(self, prev_layer, training=True):
        if training:
            return prev_layer * gpu_host(self.rng.binomial(size=(self.n_in,), p=self.p).astype(theano.config.floatX))
        else:
            return prev_layer * self.p

    def __str__(self):
        return "Dropout:%d" % self.n_in

class ActivationLayer(Layer):
    ACT_DICT = {
        'tanh': T.tanh,
        'sigmoid': T.nnet.sigmoid,
        'softmax': T.nnet.softmax,
        'relu': lambda x: x * (x > 0),
        'relu5': lambda x: x * (x > 0) + 0.5*x*(x<0)
    }
    """ Activation layer """
    def __init__(self, act):
        self.act = act

    def serialize(self):
        return [self.act]

    def unserialize(self, wt):
        self.act = wt[0]

    def forward(self, previous, training=True):
        return ActivationLayer.ACT_DICT[self.act](previous)

TanhLayer = ActivationLayer('tanh')
SigmLayer = ActivationLayer('sigmoid')
SoftMaxLayer = ActivationLayer('softmax')
ReLULayer = ActivationLayer('relu')
ReLU5Layer = ActivationLayer('relu5')

class SquaredLoss(object):
    def __init__(self, wt):
        super(SquaredLoss, self).__init__()
        self.wt = wt

    def loss(self, labels, predictions):
        diff = labels-predictions
        diff = diff*self.wt
        loss = T.sum(diff*diff)/diff.shape[0]
        return loss

class L1Reg(object):
    def __init__(self, expr):
        super(L1Reg, self).__init__()
        self.expr = expr

    def loss(self, labels, predictions):
        return T.sum(T.abs_(self.expr))/labels.shape[0]

class L2Reg(object):
    def __init__(self, expr):
        super(L2Reg, self).__init__()
        self.expr = expr

    def loss(self, labels, predictions):
        return T.sum(self.expr*self.expr)/labels.shape[0]

class SumLoss(object):
    def __init__(self, losses, wts):
        super(SumLoss, self).__init__()
        self.losses = losses
        self.wts = wts

    def loss(self, labels, predictions):
        sumloss = self.wts[0]*self.losses[0].loss(labels, predictions)
        for i in range(1, len(self.losses)):
            sumloss += self.wts[i]*self.losses[i].loss(labels, predictions)
        return sumloss

def dummytest():
    np.random.seed(10)
    N = 1000
    din = 15
    dout = 9
    FFF = np.random.randn(dout, din)
    data_in = np.random.randn(N,din)
    data_out = FFF.dot(data_in.T).T+ 0.00*np.random.randn(N, dout)

    net = NNetDyn([din,200,60,dout])

    lr = 0.08
    lr = lr/1
    lr_schedule = {
            2000: 0.5,
            3500: 0.5,
            7000: 0.5
            }

    # Train one example at a time - use high momentum
    for i in range(10*N):
        data_idx = i%N
        _din = np.expand_dims(data_in[data_idx], axis=0)
        _dout = np.expand_dims(data_out[data_idx], axis=0)
        #_din = data_in
        #_dout = data_out
        net.train(_din, _dout, lr, 0.99)
        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 200 == 0:
            print i, net.obj_matrix(data_in, data_out)

    #print FFF.T
    print 'Pred:'
    idx = 3
    pred =  net.fwd_single(data_in[idx])
    F, f = net.getF(data_in[idx] + 0.1*np.random.randn(din))
    print 'F:', F.T
    print 'f;',f
    pred2 = (F.dot(data_in[idx])+f)
    print pred
    print pred2
    print data_out[idx]
    print FFF.dot(data_in[idx])

def test_whiten():
    w = WhitenLayer()
    data = np.random.randn(30,5)
    w.generate_weights(data)
    whitened = w.forward(data)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    #get_net()
    #train_dyn()
    test_whiten()
