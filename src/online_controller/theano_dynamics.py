import time
import numpy as np
import sys
import cPickle
import theano
import theano.tensor as T
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
    def __init__(self, layers, loss_wt, init_funcs=True, weight_decay=0.0):
        self.loss = SquaredLoss(loss_wt)
        self.params = []
        self.nparams = 0
        self.layers = layers
        self.weight_decay = weight_decay
        if init_funcs:
            self.init_funcs()

def init_funcs(self):
        self.params = []
        self.nparams = 0
        for layer in self.layers:
            self.params.extend(layer.params())
            self.nparams += layer.n_params()

        jac_data = T.vector('jacdata')
        jac_lbl = T.vector('jaclbl')
        net_out, obj = self.fwd(jac_data, jac_lbl)
        self.net_out_vec = theano.function(inputs=[jac_data], outputs=net_out)
        self.out_shape = theano.function(inputs=[jac_data], outputs=net_out.shape)
        data_grad = theano.gradient.jacobian(net_out, jac_data)
        self.jac = theano.function(inputs=[jac_data], outputs=data_grad)

        data = T.matrix('data')
        lbl = T.matrix('lbl')
        # Cannot be serialized
        net_out = self.fwd(data)
        jacobian = theano.gradient.jacobian(net_out, data)
        obj = self.loss.loss(labels, net_out, )
        self.obj = theano.function(inputs=[data, lbl], outputs=obj, on_unused_input='warn')
        self.train_sgd = train_gd_momentum(obj, self.params, [data, lbl], scl=self.nparam


    def serialize(self):
        wts = []
        for layer in self.layers:
            wts.append(layer)
        metadata = {
            'time': time.time(),
            'loss': self.loss,
            'weight_decay': self.weight_decay
        }
        return (metadata, wts)

    def unserialize(self, net):
        metadata = net[0]
        wts = net[1]
        self.loss = metadata['loss']
        self.loss.wt[7:14] = 2.0
        self.weight_decay = metadata['weight_decay']
        self.layers = [None]*len(wts)
        for i in range(len(wts)):
            self.layers[i] = wts[i]

    def fwd(self, data, labels):
        net_out = data
        for layer in self.layers:
            net_out = layer.forward(net_out)
        obj = self.loss.loss(labels, net_out)
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
        pass

    def forward(self, prev_layer):
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
        return [self.sig, self.mu, self.normalize]

    def unserialize(self, wts):
        self.sig = wts[0]
        self.mu = wts[1]
        self.normalize = self.normalize

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

    def forward(self, prev_layer):
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

    def forward(self, previous):
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

class SparsityLoss(object):
    def __init__(self, expr):
        super(SparsityLoss, self).__init__()
        self.expr = expr

    def loss(self, labels, predictions):
        return T.sum(self.expr)

class SumLoss(object):
    def __init__(self, losses):
        super(SumLoss, self).__init__()
        self.losses = losses

    def loss(self, labels, predictions):
        sumloss = self.losses[0].loss(labels, predictions)
        for i in range(1, len(self.losses)):
            sumloss += self.losses[i].loss(labels, predictions)
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

def train_dyn():
    fname = 'norm_net.pkl'
    #np.random.seed(10)
    np.set_printoptions(suppress=True)
    import scipy.io
    data = scipy.io.loadmat('dyndata.mat')
    din = 46
    dout = 39
    data_in = data['train_data'].T.astype(np.float32)
    print 'Data shape:', data_in.shape
    N = data_in.shape[0]
    print 'Loaded %d training examples' % N
    data_out = data['train_lbl'].T.astype(np.float32)
    #shuffle
    perm = np.random.permutation(N)
    data_in = data_in[perm]
    data_out = data_out[perm]

    try:
        net = load_net(fname)
        print 'Loaded net!'
    except IOError:
        print 'Initializing new network!'
        wt = np.ones(39).astype(np.float32)
        wt[0:7] = 5.0
        wt[7:14] = 2.0
        norm1 = NormalizeLayer()
        norm1.generate_weights(data_in)
        #norm2 = NormalizeLayer()
        #norm2.generate_weights(data_out, reverse=True)
        net = NNetDyn([norm1, FFIPLayer(46,100), ReLULayer, FFIPLayer(100,50) , ReLULayer, FFIPLayer(50,39)], wt, weight_decay=0.0001)
        import pdb; pdb.set_trace()


    for idx in [25]:
        pred =  net.fwd_single(data_in[idx])
        F, f = net.getF(data_in[idx] + 0.01*np.random.randn(din).astype(np.float32))
        print F
        print f
        pred2 = (F.dot(data_in[idx])+f)
        print 'in:',data_in[idx]
        print 'net:',pred
        print 'tay:',pred2
        print 'lbl:',data_out[idx]
        import pdb; pdb.set_trace()

    # Train one example at a time - use high momentum
    #lr = 20.0
    bsize = 50
    lr = 10.0/bsize
    lr_schedule = {
            1000000: 0.5,
            2000000: 0.5,
            5000000: 0.2
            }

    epochs = 0
    for i in range(10*1000*1000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            perm = np.random.permutation(N)
            data_in = data_in[perm]
            data_out = data_out[perm]
            continue
        #_din = np.expand_dims(data_in[data_idx], axis=0)
        #_dout = np.expand_dims(data_out[data_idx], axis=0)
        _din = data_in[bstart:bend]
        _dout = data_out[bstart:bend]
        #_din = data_in
        #_dout = data_out
        net.train(_din, _dout, lr, 0.90)
        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 5000 == 0:
            print i, net.obj_matrix(data_in, data_out)
            sys.stdout.flush()
        if i % 50000 == 0:
            print 'Dumping weights'
            dump_net(fname, net)

    for idx in [1,2, 90]:
        pred =  net.fwd_single(data_in[idx])
        F, f = net.getF(data_in[idx] + 0.1*np.random.randn(din).astype(np.float32))
        print F
        print f
        pred2 = (F.dot(data_in[idx])+f)
        print 'in:',data_in[idx]
        print 'net:',pred
        print 'tay:',pred2
        print 'lbl:',data_out[idx]



def get_net():
    wt = np.ones(39).astype(np.float32)
    #net = NNetDyn([46,200,100,100,39], wt)
    with open('dyn_net_lin.pkl', 'r') as f:
        wts = cPickle.load(f)
        #net = NNetDyn([46,200,TanhLayer,300,ReLULayer,100,ReLULayer,39], wt)
        net = NNetDyn([46,39], wt)
        net.set_weights(wts)
    return net

def test_norm():
    data = scipy.io.loadmat('dyndata.mat')
    data_in = data['train_data'].T.astype(np.float32)
    print data_in.shape
    norm = NormalizeLayer()
    norm.generate_weights(data_in)
    norm_data = norm.forward(data_in)
    norm.reverse = True
    norm_data_rev = norm.forward(norm_data)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    #get_net()
    train_dyn()
    #test_norm()
