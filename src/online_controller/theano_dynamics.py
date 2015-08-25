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

def load_rec_net(fname, dX, dU):
    with open(fname, 'r') as pklfile:
        layers = cPickle.load(pklfile)
    net = NNetRecursive(dX, dU, 1, layers)
    return net

class NNetDyn(object):
    def __init__(self, layers, loss_wt, post_layer=None, init_funcs=True,weight_decay=0.0, layer_reg = [], recurse=None):
        self.params = []
        self.loss_wt = loss_wt
        self.layer_reg = layer_reg
        self.nparams = 0
        self.layers = layers
        self.post_layer = post_layer
        self.weight_decay = weight_decay
        self.recurse = recurse
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
        self.loss = SquaredLoss(self.loss_wt)

        net_out, layers_out = self.fwd(data, training=True)
        obj = self.loss.loss(lbl, net_out)
        self.obj = theano.function(inputs=[data, lbl], outputs=obj, on_unused_input='warn')
        self.train_sgd = train_gd_momentum(obj, self.params, [data, lbl], scl=1.0, weight_decay=self.weight_decay)

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
        self.loss_wt = metadata['loss_wt']
        self.layer_reg = metadata.get('layer_reg', [])
        self.post_layer = metadata.get('post_layer', None)
        self.weight_decay = metadata['weight_decay']
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

    def getF_prevstate(self, prevxu, xu):
        """ Return F, f - 1st order Taylor expansion of network around xu
            For the f(xt-1, ut-1, xt, ut) -> xt+1 network
         """
        dXU = xu.shape[0]
        xuxu = np.concatenate(prevxu, xu)
        F_full = self.jac(xuxu)
        f_full = -F.dot(xuxu)+self.fwd_single(xuxu)

        F = F[:,dXU:2*dXU] 
        f = f_full + F[:,0:dXU].dot(prevxu)
        return F, f

    def __str__(self):
        return str(self.layers)

class NNetRecursive(object):
    @staticmethod
    def prepare_data(data, lbl, clip, dX, dU, lookahead):
        N = data.shape[0]
        Xs = data[:,:dX]
        Us = data[:,dX:dX+dU]

        data_X = []
        data_U = []
        tgt_X = []
        for n in range(N):
            x = Xs[n]
            us = []
            tgt_x = []

            fail = False
            for t in range(lookahead):
                #print 'clip:',clip[n+t], '>', clip[n+t+1]
                # Doing clip this way loses 1 data point per trial (which is stored in lbl)
                if (n+t>=N) or (t>0 and clip[n+t] == 0): # Hit a boundary
                    fail = True
                    break
                us.append(Us[n+t])
                tgt_x.append(lbl[n+t])

            if not fail:
                us = np.concatenate(us)
                tgt_x = np.concatenate(tgt_x)

                data_X.append(x)
                data_U.append(us)
                tgt_X.append(tgt_x)

        data_X = np.c_[data_X]
        data_U = np.c_[data_U]
        tgt_X = np.c_[tgt_X]
        return data_X, data_U, tgt_X

    # Train a feedforward net to predict multiple timesteps
    def __init__(self, dX, dU, T, layers, weight_decay=0, init_funcs=True):
        self.dX = dX
        self.dU = dU
        self.layers = layers
        self.T = T
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
        X = T.matrix('mX')
        U = T.matrix('mU')
        Xtgt = T.matrix('mXtgt')
        self.loss = SquaredLoss()

        obj = self.obj_rec_symbolic(X, U, Xtgt, training=True)
        self.obj = theano.function(inputs=[X, U, Xtgt], outputs=obj, on_unused_input='warn')
        self.train_sgd = train_gd_momentum(obj, self.params, [X, U, Xtgt], scl=1.0, weight_decay=self.weight_decay)

        # Set up vector functions
        X = T.vector('vX')
        U = T.vector('vU')
        Xtgt = T.vector('vXtgt')
        net_out = self.fwd_symbolic_single(X, U, training=False)
        self.net_out_vec = theano.function(inputs=[X, U], outputs=net_out)
        self.out_shape = theano.function(inputs=[X, U], outputs=net_out.shape)

        x_grad = theano.gradient.jacobian(net_out, X)
        u_grad = theano.gradient.jacobian(net_out, U)
        data_grad = T.concatenate([x_grad, u_grad], axis=1)
        self.jac = theano.function(inputs=[X, U], outputs=data_grad)

    def fwd_symbolic_single(self, x, u, training=True):
        xu = T.concatenate([x, u])
        net_out = xu
        for layer in self.layers:
            layer.set_input_data(xu)
            net_out = layer.forward(net_out, training=training)
        return net_out

    def fwd_symbolic(self, x, u, training=True):
        xu = T.concatenate([x, u], axis=1)
        net_out = xu
        for layer in self.layers:
            layer.set_input_data(xu)
            net_out = layer.forward(net_out, training=training)
        return net_out

    def getF(self, xu):
        """ Return F, f - 1st order Taylor expansion of network around xu """
        x = xu[0:self.dX]
        u = xu[self.dX:self.dX+self.dU]
        return self.getFxu(x,u)

    def getFxu(self, x, u):
        """ Return F, f - 1st order Taylor expansion of network around xu """
        F = self.jac(x, u)
        f = -F.dot(np.concatenate([x, u]))+self.net_out_vec(x, u)
        return F, f

    def obj_rec_symbolic(self, X, U, Xtgt, training=True):
        prevX = X
        loss = None
        for t in range(self.T):
            u = U[:,t*self.dU:(t+1)*self.dU]
            lbl = Xtgt[:,t*self.dX:(t+1)*self.dX]
            net_out = self.fwd_symbolic(prevX, u, training=training)
            l = self.loss.loss(net_out, lbl)
            if loss is None:
                loss = l
            else:
                loss += l
            prevX = net_out
        return loss

    def train(self, xs, us, xtgt, lr=0.1, momentum=0.9):
        return self.train_sgd(xs, us, xtgt, lr, momentum)

    def train_single(self, xu, tgt, lr, momentum):
        """Run SGD on a single (vector) data point """
        xu = np.expand_dims(xu, axis=0).astype(np.float32)
        x = xu[:,0:self.dX]
        u = xu[:,self.dX:self.dX+self.dU]
        tgt = np.expand_dims(tgt, axis=0).astype(np.float32)
        return self.train_sgd(x, u, tgt, lr, momentum)[0]

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
        #self.idxprevu = slice(14,21)
        self.idxeepos = slice(14,23)
        self.idxeevel = slice(23,32)
        self.idxu = slice(32,39)

    def forward(self, prev_layer, training=True):
        if training:
            jnt_accel = prev_layer[:,:7]
            ee_accel = prev_layer[:,7:16]
            t = self.t
            jnt_pos = self.input_data[:,self.idxpos] + t*self.input_data[:,self.idxvel] + t*t*jnt_accel
            jnt_vel = self.input_data[:,self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[:,self.idxeepos]+ t*self.input_data[:,self.idxeevel] + t*t*ee_accel
            ee_vel = self.input_data[:,self.idxeevel] + t*ee_accel
            return T.concatenate([jnt_pos, jnt_vel, ee_pos, ee_vel], axis=1)
        else:
            jnt_accel = prev_layer[:7]
            ee_accel = prev_layer[7:16]
            t = self.t
            jnt_pos = self.input_data[self.idxpos] + t*self.input_data[self.idxvel] + t*t*jnt_accel
            jnt_vel = self.input_data[self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[self.idxeepos]+ t*self.input_data[self.idxeevel] + t*t*ee_accel
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
        self.idxft = slice(14,20)
        self.idxeepos = slice(20,29)
        self.idxeevel = slice(29,38)
        self.idxu = slice(38,45)

    def forward(self, prev_layer, training=True):
        if training:
            jnt_accel = prev_layer[:,:7]
            ee_accel = prev_layer[:,7:16]
            nextft = prev_layer[:,16:22]
            t = self.t
            jnt_pos = self.input_data[:,self.idxpos] + t*self.input_data[:,self.idxvel] + t*t*jnt_accel
            jnt_vel = self.input_data[:,self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[:,self.idxeepos]+ t*self.input_data[:,self.idxeevel] + t*t*ee_accel
            ee_vel = self.input_data[:,self.idxeevel] + t*ee_accel
            nextft = nextft + self.input_data[:,self.idxft]
            return T.concatenate([jnt_pos, jnt_vel, nextft, ee_pos, ee_vel], axis=1)
        else:
            jnt_accel = prev_layer[:7]
            ee_accel = prev_layer[7:16]
            nextft = prev_layer[16:22]
            t = self.t
            jnt_pos = self.input_data[self.idxpos] + t*self.input_data[self.idxvel] + t*t*jnt_accel
            jnt_vel = self.input_data[self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[self.idxeepos]+ t*self.input_data[self.idxeevel] + t*t*ee_accel
            ee_vel = self.input_data[self.idxeevel] + t*ee_accel
            nextft = nextft + self.input_data[self.idxft]
            return T.concatenate([jnt_pos, jnt_vel, nextft, ee_pos, ee_vel])

    def __str__(self):
        return "AccelLayerFT"


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
            jnt_pos = self.input_data[:,self.idxpos] + t*self.input_data[:,self.idxvel] + t*t*jnt_accel
            jnt_vel = self.input_data[:,self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[:,self.idxeepos]+ t*self.input_data[:,self.idxeevel] + t*t*ee_accel
            ee_vel = self.input_data[:,self.idxeevel] + t*ee_accel
            return T.concatenate([jnt_pos, jnt_vel, ee_pos, ee_vel], axis=1)
        else:
            jnt_accel = prev_layer[:7]
            ee_accel = prev_layer[7:13]
            t = self.t
            jnt_pos = self.input_data[self.idxpos] + t*self.input_data[self.idxvel] + t*t*jnt_accel
            jnt_vel = self.input_data[self.idxvel] + t*jnt_accel
            ee_pos = self.input_data[self.idxeepos]+ t*self.input_data[self.idxeevel] + t*t*ee_accel
            ee_vel = self.input_data[self.idxeevel] + t*ee_accel
            return T.concatenate([jnt_pos, jnt_vel, ee_pos, ee_vel])

    def __str__(self):
        return "AccelLayer"

class GatedLayer(Layer):
    """ Gated Layer """
    def __init__(self, layers, din, dout):
        self.layers = layers
        self.ffip = FFIPLayer(din, dout)

    def forward(self, prev_layer, training=True):
        ip = self.ffip.forward(prev_layer, training=training)
        gates = prev_layer
        for layer in self.layers:
            gates = layer.forward(gates, training=training)
        return gates * ip

    def __str__(self):
        return "GatedLayer"

class ControlAffine(Layer):
    """ ControlAffine Layer """
    def __init__(self, layers, dx):
        self.layers = layers
        self.ffip = FFIPLayer(din, dout)

    def forward(self, prev_layer, training=True):
        pass

    def __str__(self):
        return "ControlAffine"

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
        'softplus': T.nnet.softplus,
        'relu': lambda x: x * (x > 0),
        'relu5': lambda x: x * (x > 0) + 0.5*x*(x<0),
        'relunop': lambda x: x * (x > 0) + 1.0*x*(x<0) #Debugging

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
SoftPlusLayer = ActivationLayer('softplus')
ReLU5Layer = ActivationLayer('relu5')
ReLUNopLayer = ActivationLayer('relunop')

class SquaredLoss(object):
    def __init__(self, wt=None):
        super(SquaredLoss, self).__init__()
        self.wt = wt

    def loss(self, labels, predictions):
        diff = labels-predictions
        if self.wt is not None:
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
