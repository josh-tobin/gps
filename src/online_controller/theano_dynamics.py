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
    def __init__(self, layers, loss_wt, post_layer=None, init_funcs=True, weight_decay=0.0):
        self.loss = SquaredLoss(loss_wt)
        self.params = []
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

        data = T.matrix('data')
        lbl = T.matrix('lbl')

        # Cannot be serialized
        net_out, obj = self.fwd(data, lbl, training=True)
        self.obj = theano.function(inputs=[data, lbl], outputs=obj, on_unused_input='warn')
        self.train_sgd = train_gd_momentum(obj, self.params, [data, lbl], scl=self.nparams, weight_decay=self.weight_decay)

        jac_data = T.vector('jacdata')
        jac_lbl = T.vector('jaclbl')
        net_out, obj = self.fwd(jac_data, jac_lbl, training=False, post_layer=True)
        self.net_out_vec = theano.function(inputs=[jac_data], outputs=net_out)
        self.out_shape = theano.function(inputs=[jac_data], outputs=net_out.shape)
        data_grad = theano.gradient.jacobian(net_out, jac_data)
        self.jac = theano.function(inputs=[jac_data], outputs=data_grad)

    def serialize(self):
        wts = []
        for layer in self.layers:
            wts.append(layer)
        metadata = {
            'time': time.time(),
            'losswt': self.loss.wt,
            'weight_decay': self.weight_decay,
            'post_layer': self.post_layer
        }
        return (metadata, wts)

    def unserialize(self, net):
        metadata = net[0]
        wts = net[1]
        #self.loss = metadata['loss']
        self.loss.wt = metadata['losswt']

        self.post_layer = metadata.get('post_layer', None)

        self.loss.wt[0:7] = 5.0
        self.loss.wt[7:14] = 2.0
        #self.loss.wt[7:14] = 2.0
        self.weight_decay = metadata['weight_decay']
        self.layers = [None]*len(wts)
        for i in range(len(wts)):
            self.layers[i] = wts[i]

    def fwd(self, data, labels, training=True, post_layer=True):
        net_out = data
        for layer in self.layers:
            net_out = layer.forward(net_out, training=training)
        obj = self.loss.loss(labels, net_out)
        if post_layer and self.post_layer:
            net_out = self.post_layer.forward(net_out, training=training)
        return net_out, obj 

    def fwd_single(self, xu):
        #xu = np.concatenate((xu, ONE))
        return self.net_out_vec(xu)

    def train(self, xu, tgt, lr, momentum):
        """ Update dynamics via SGD """
        #xu = np.c_[xu, np.ones((xu.shape[0],1))]
        self.train_sgd(xu, tgt, lr, momentum)

    def obj_matrix(self, xu, tgt):
        """ Update dynamics via SGD """
        #xu = np.c_[xu, np.ones((xu.shape[0],1))]
        return self.obj(xu, tgt)

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


class NormalizeLayer(object):
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

    def params(self):
        """ Return a list of trainable parameters """
        return []
    
    def n_params(self):
        return 0

    def serialize(self):
        return [self.sig, self.mean, self.reverse]

    def unserialize(self, wts):
        self.sig = wts[0]
        self.mean = wts[1]
        self.reverse = self.reverse

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())

class WhitenLayer(object):
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

    def params(self):
        """ Return a list of trainable parameters """
        return []
    
    def n_params(self):
        return 0

    def serialize(self):
        return [self.sig, self.mean, self.reverse]

    def unserialize(self, wts):
        self.sig = wts[0]
        self.mean = wts[1]
        self.reverse = self.reverse

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())


class FFIPLayer(object):
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

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())
        #return "W:%s" % (self.w.get_value())

class DropoutLayer(object):
    """ Feedforward inner product layer """
    def __init__(self, n_in, rngseed=10):
        self.n_in = n_in
        self.rng = RandomStreams(seed=rngseed)

    def forward(self, prev_layer, training=True):
        if training:
            return prev_layer * gpu_host(self.rng.binomial(size=(self.n_in,),p=0.5).astype(theano.config.floatX))
        else:
            return prev_layer * 0.5

    def params(self):
        """ Return a list of trainable parameters """
        return []
    
    def n_params(self):
        return 0

    def serialize(self):
        return []

    def unserialize(self, wts):
        pass

    def __str__(self):
        return "Dropout:%d" % self.n_in

class ActivationLayer(object):
    ACT_DICT = {
        'tanh': T.tanh,
        'sigmoid': T.nnet.sigmoid,
        'softmax': T.nnet.softmax,
        'relu': lambda x: x * (x > 0) + (0.99*x * (x<0))
    }
    """ Activation layer """
    def __init__(self, act):
        self.act = act

    def serialize(self):
        return [self.act]

    def unserialize(self, wt):
        self.act = wt[0]

    def params(self):
        return []

    def n_params(self):
        return 0

    def forward(self, previous, training=True):
        return ActivationLayer.ACT_DICT[self.act](previous)

TanhLayer = ActivationLayer('tanh')
SigmLayer = ActivationLayer('sigmoid')
SoftMaxLayer = ActivationLayer('softmax')
ReLULayer = ActivationLayer('relu')

class SquaredLoss(object):
    def __init__(self, wt):
        super(SquaredLoss, self).__init__()
        self.wt = wt

    def loss(self, labels, predictions):
        diff = labels-predictions
        diff = diff*self.wt
        loss = T.sum(diff*diff)/diff.shape[0]
        return loss

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
