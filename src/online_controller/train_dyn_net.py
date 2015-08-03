from theano_dynamics import *
import numpy as np
import sys

def get_data():
    data = scipy.io.loadmat('dyndata.mat')
    data_in = data['train_data'].T.astype(np.float32)
    #data_in = np.c_[data_in[:,0:27], data_in[:,45:52]]
    #data_in = np.c_[data_in[:,0:21], data_in[:,39:46]]
    print 'Data shape:', data_in.shape
    data_out = data['train_lbl'].T.astype(np.float32)

    data2 = scipy.io.loadmat('dyn_data_Aug2.mat')
    X = data2['data'].T # N x Dx+Du
    N = X.shape[0]
    test_data = []
    test_lbl = []
    for n in range(N/100):
        for t in range(99):
            x = X[t+100*n,:39]
            u = X[t+100*n,39:46]
            x_p1 = X[t+1+100*n,:39]
            xu = np.r_[x,u]
            test_data.append(xu)
            test_lbl.append(x_p1)
    test_data = np.array(test_data).astype(np.float32)
    test_lbl = np.array(test_lbl).astype(np.float32)
    return data_in, data_out, test_data, test_lbl

def train_dyn():
    fname = 'trap_contact_%s.pkl' % sys.argv[1]
    #np.random.seed(10)
    np.set_printoptions(suppress=True)
    import scipy.io
    din = 46#28
    dout = 39#21
    data_in, data_out, test_data, test_lbl = get_data()
    N = data_in.shape[0]
    #data_out = data_out[:,0:27]
    #data_out = data_out[:,0:21]

    #shuffle
    #perm = np.random.permutation(N)
    #data_in = data_in[perm]
    #data_out = data_out[perm]

    wt = np.ones(39).astype(np.float32)
    wt[0:7] = 5.0
    wt[7:14] = 2.0
    #norm1 = NormalizeLayer()
    norm1 = WhitenLayer()
    norm1.generate_weights(data_in)

    """
    norm2 = NormalizeLayer()
    norm2.generate_weights(data_out)
    data_out_normed = norm2.forward(data_out)
    norm2.generate_weights(data_out, reverse=True)
    data_out_unnorm = data_out
    data_out = data_out_normed
    """
    #net = NNetDyn([norm1, FFIPLayer(din,200), TanhLayer, FFIPLayer(200,200) , ReLULayer, FFIPLayer(200,dout)], wt, weight_decay=0.0001)
    #net = NNetDyn([norm1, FFIPLayer(din,100), TanhLayer, FFIPLayer(100,100), ReLULayer, FFIPLayer(100,dout)], wt, weight_decay=0.0005)
    #net = NNetDyn([norm1, FFIPLayer(din,100), TanhLayer, FFIPLayer(100,80), ReLULayer, FFIPLayer(80,dout)], wt, weight_decay=0.0001, sparsity_wt = 0.1)
    net = NNetDyn([norm1, FFIPLayer(din,40), ReLULayer, FFIPLayer(40,dout)], wt, weight_decay=0.0001)
    #net = NNetDyn([norm1, FFIPLayer(din,dout)], wt, weight_decay=0.0001) # loss ~0.13

    try:
        net = load_net(fname)
        print 'Loaded net!'
    except IOError:
        print 'Initializing new network!'

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

    bsize = 1
    lr = 5.0/bsize
    lr_schedule = {
            2000000: 0.2,
            4000000: 0.2,
            6000000: 0.2,
            8000000: 0.2,
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
            print 'Train:',i, net.obj_matrix(data_in, data_out), ' // Test :',i, net.obj_matrix(test_data, test_lbl)
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



def get_net(name):
    net = load_net(name)
    print net.getF(np.zeros(46).astype(np.float32))
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
