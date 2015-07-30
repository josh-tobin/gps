import numpy as np
import sys
import cPickle
import theano
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as tcuda
ONE = np.array([1])

NN_INIT = 0.05

gpu_host = tcuda.gpu_from_host

class NNetDyn(object):
    def __init__(self, dims, loss_wt):
        data = T.matrix('data')
        lbl = T.matrix('lbl')

        self.loss = SquaredLoss(loss_wt)
        self.layers = []
        self.params = []
        self.nparams = 0

        #dims[0] += 1
        for i in range(len(dims)-1):
            din = dims[i]
            dout = dims[i+1]
            if type(dout) == ActivationLayer:
                self.layers.append(dout)
                dims[i+1] = din
                i += 1
            else:
                new_layer = FFIPLayer(din, dout)
                self.layers.append(new_layer)
                self.params.extend(new_layer.params())
                self.nparams += new_layer.n_params()
        print self.layers
        net_out, obj = self.fwd(data, lbl)
        self.obj = theano.function(inputs=[data, lbl], outputs=obj, on_unused_input='warn')

        self.train_sgd = train_gd_momentum(obj, self.params, [data, lbl], scl=self.nparams)

        jac_data = T.vector('jacdata')
        jac_lbl = T.vector('jaclbl')
        net_out, obj = self.fwd(jac_data, jac_lbl)
        self.net_out_vec = theano.function(inputs=[jac_data], outputs=net_out)
        self.out_shape = theano.function(inputs=[jac_data], outputs=net_out.shape)
        data_grad = theano.gradient.jacobian(net_out, jac_data)
        self.jac = theano.function(inputs=[jac_data], outputs=data_grad)

    def get_weights(self):
        wts = []
        for layer in self.layers:
            wts.append(layer.get_weights())
        return wts

    def set_weights(self, wts):
        for i in range(len(wts)):
            self.layers[i].set_weights(wts[i])

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


def train_gd_momentum(obj, params, args, scl=1.0):
    obj = obj
    scl = float(scl)
    gradients = T.grad(obj, params)
    eta = T.scalar('lr')
    momentum = T.scalar('momentum')
    momentums = [theano.shared(np.copy(param.get_value())) for param in params]
    updates = []
    for i in range(len(gradients)):
        update_gradient = (gradients[i])+momentum*momentums[i]
        updates.append((params[i], gpu_host(params[i]-(eta/scl)*update_gradient)))
        updates.append((momentums[i], gpu_host(update_gradient)))
    train = theano.function(
        inputs=args+[eta, momentum],
        outputs=[obj],
        updates=updates
    )
    return train


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

    def get_weights(self):
        return [self.w.get_value(), self.b.get_value()]

    def set_weights(self, wts):
        self.w.set_value(wts[0])
        self.b.set_value(wts[1])

    def __str__(self):
        return "W:%s; b:%s" % (self.w.get_value(), self.b.get_value())
        #return "W:%s" % (self.w.get_value())

class ActivationLayer(object):
    """ Activation layer """
    def __init__(self, act):
        self.act = act

    def get_weights(self):
        return None

    def set_weights(self, wt):
        pass

    def forward(self, previous):
        return self.act(previous)

TanhLayer = ActivationLayer(T.tanh)
SigmLayer = ActivationLayer(T.nnet.sigmoid)
SoftMaxLayer = ActivationLayer(T.nnet.softmax)
ReLULayer = ActivationLayer(lambda x: x * (x > 0) + (0.99*x * (x<0)))

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

def train_dyn():
    fname = 'dyn_net_air2.pkl'
    np.random.seed(10)
    np.set_printoptions(suppress=True)
    import scipy.io
    data = scipy.io.loadmat('dyndata.mat')
    din = 46
    dout = 39
    data_in = data['train_data'].T.astype(np.float32)
    N = data_in.shape[0]
    print 'Loaded %d training examples' % N
    data_out = data['train_lbl'].T.astype(np.float32)
    wt = np.ones(39).astype(np.float32)
    wt[0:7] = 5.0
    wt[0] = 10.0
    wt[7:14] = 2.0
    net = NNetDyn([din,200, ReLULayer, 300, ReLULayer, 100, ReLULayer, dout], wt)

    #shuffle
    perm = np.random.permutation(N)
    data_in = data_in[perm]
    data_out = data_out[perm]

    try:
        with open(fname, 'r') as f:
            wts = cPickle.load(f)
            net.set_weights(wts)
    except IOError:
        print 'Initializing new network!'


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

    # Train one example at a time - use high momentum
    #lr = 20.0
    bsize = 50
    lr = 50.0/bsize
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
        if i % 1000 == 0:
            print i, net.obj_matrix(data_in, data_out)
            sys.stdout.flush()
        if i % 100000 == 0:
            print 'Dumping weights'
            with open(fname, 'w') as f:
                cPickle.dump(net.get_weights(), f)

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

    with open(fname, 'w') as f:
        cPickle.dump(net.get_weights(), f)

def get_net():
    wt = np.ones(39).astype(np.float32)
    #net = NNetDyn([46,200,100,100,39], wt)
    with open('dyn_net_trap.pkl', 'r') as f:
        wts = cPickle.load(f)

        net = NNetDyn([46,200,100,100,39], wt)
        net.set_weights(wts)
    return net

if __name__ == "__main__":
    #get_net()
    train_dyn()
