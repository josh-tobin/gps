from theano_dynamics import *
import numpy as np
import sys
import os

DATA_DIR = 'data'
NET_DIR = 'net'

def load_matfile(matfile, remove_ft=False):
    data = scipy.io.loadmat(matfile)
    adjust_eetgt = False
    try:
        dat = data['data'].astype(np.float32)
        lbl = data['label'].astype(np.float32)
        if 'eetgt' in data:
            adjust_eetgt = True
            eetgt = data['eetgt'].astype(np.float32)
            eetgt_idx = data['eetgt_idx'][0]
            eetgt_idx = slice(eetgt_idx[0], eetgt_idx[1])
    except KeyError:
        dat = data['train_data'].T.astype(np.float32)
        lbl = data['train_lbl'].T.astype(np.float32)

    if dat.shape[1] > dat.shape[0]:
        dat = dat.T
    if lbl.shape[1] > lbl.shape[0]:
        lbl = lbl.T

    if adjust_eetgt:
        dat[:,eetgt_idx] += eetgt
        lbl[:,eetgt_idx] += eetgt

    if remove_ft:
        if 'ft_idx' in data:
            ft_idx = data['ft_idx'][0]
            dat = np.c_[dat[:,:ft_idx[0]], dat[:,ft_idx[1]:]]
            lbl = np.c_[lbl[:,:ft_idx[0]], lbl[:,ft_idx[1]:]]
        else:
            print 'No FT idx for ', matfile

    return dat, lbl
        

def get_data(train, test, shuffle_test=True, remove_ft=True):
    train_data = [] 
    train_label = []
    for matfile in train:
        data_in, data_out = load_matfile(os.path.join(DATA_DIR,matfile), remove_ft=remove_ft)
        train_data.append(data_in)
        #data_in = np.c_[data_in[:,0:27], data_in[:,45:52]]
        #data_in = np.c_[data_in[:,0:21], data_in[:,39:46]]
        train_label.append(data_out)
        print 'Train data: Loaded %s. Shape: %s' % (matfile, data_in.shape)
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)

    test_data = [] 
    test_label = []
    for matfile in test:
        data_in, data_out = load_matfile(os.path.join(DATA_DIR,matfile), remove_ft=remove_ft)
        test_data.append(data_in)
        test_label.append(data_out)
        print 'Test data: Loaded %s. Shape: %s' % (matfile, data_in.shape)
    test_data = np.concatenate(test_data)
    test_label = np.concatenate(test_label)

    #shuffle
    perm = np.random.permutation(test_data.shape[0])
    test_data = test_data[perm]
    test_label = test_label[perm]
    """
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
    scipy.io.savemat('dyndata_plane_extra.mat', {'data':test_data, 'label':test_lbl})
    """
    test2train = 0
    train_data = np.r_[train_data, test_data[:test2train,:]]
    #train_data = np.r_[data_in]
    train_lbl = np.r_[train_label, test_label[:test2train,:]]
    #train_lbl = np.r_[data_out]
    test_data = test_data[test2train:,:]
    test_lbl = test_label[test2train:,:]

    #train_data = train_data[:200,:]
    #train_lbl = train_lbl[:200,:]
    return train_data, train_lbl, test_data, test_lbl

def train_dyn():
    fname = os.path.join(NET_DIR, '%s.pkl' % sys.argv[1])
    #np.random.seed(10)
    np.set_printoptions(suppress=True)
    import scipy.io
    din = 46
    dout = 39
    data_in, data_out, test_data, test_lbl = get_data(['dyndata_armwave', 'dyndata_armwave_2'],['dyndata_plane_relu'], remove_ft=True)
    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_plane_ft', 'dyndata_plane_ft_2'],['dyndata_plane_ft_2'], remove_ft=True, ee_tgt_adjust=None)

    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_trap_sim'],['dyndata_trap_sim'], remove_ft=False, ee_tgt_adjust=slice(21,30))
    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_trap', 'dyndata_trapskid'],['dyndata_trap2'], remove_ft=False, ee_tgt_adjust=None)

    #din = 33
    #dout = 26
    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_mjc', 'dyndata_mjc_air', 'dyndata_mjc_expr'],['dyndata_mjc_test'], remove_ft=False)

    N = data_in.shape[0]
    print data_in.shape
    print data_out.shape
    print test_data.shape
    #data_out = data_out[:,0:27]
    #data_out = data_out[:,0:21]

    wt = np.ones(dout).astype(np.float32)
    #wt[0:7] = 5.0
    #wt[7:14] = 2.0
    norm1 = NormalizeLayer()
    #norm1 = WhitenLayer()
    norm1.generate_weights(data_in)

    """
    norm2 = NormalizeLayer()
    norm2.generate_weights(data_out)
    data_out_normed = norm2.forward(data_out)
    norm2.generate_weights(data_out, reverse=True)
    data_out_unnorm = data_out
    data_out = data_out_normed
    """
    #net = NNetDyn([FFIPLayer(din, 200), DropoutLayer(200), ReLULayer, FFIPLayer(200, 200), DropoutLayer(200), ReLULayer, FFIPLayer(200,22), AccelLayerFT()], wt)
    #net = NNetDyn([FFIPLayer(din, 40), ReLULayer, FFIPLayer(40,13), AccelLayerMJC()], wt, weight_decay=0.0001)
    #net = NNetDyn([FFIPLayer(din, 70), DropoutLayer(70), ReLULayer, FFIPLayer(70,50), DropoutLayer(50, p=0.7), ReLULayer, FFIPLayer(50,16), AccelLayer()], wt)

    #net = NNetDyn([norm1, FFIPLayer(din, 50), ReLU5Layer, FFIPLayer(50,16), AccelLayer()], wt)
    net = NNetDyn([norm1, FFIPLayer(din, 60), ReLU5Layer, FFIPLayer(60,dout)], wt)
    net = NNetDyn([norm1, FFIPLayer(din, 60), ReLU5Layer, FFIPLayer(60, 60), ReLU5Layer, FFIPLayer(60,dout)], wt)
    #net = NNetDyn([FFIPLayer(din,dout)], wt, weight_decay=0.0000) # loss ~0.13

    try:
        net = load_net(fname)
        print 'Loaded net!'
    except IOError:
        print 'Initializing new network!'

    for idx in [25]:#, 325, 625]:
        pred =  net.fwd_single(data_in[idx])
        print np.linalg.norm(data_in[idx,21:24]-data_in[idx,24:27])
        print np.linalg.norm(data_in[idx,21:24]-data_in[idx,27:30])
        print np.linalg.norm(data_in[idx,24:27]-data_in[idx,27:30])

        F, f = net.getF(data_in[idx] + 0.01*np.random.randn(din).astype(np.float32))
        getFtrain = lambda idx: net.getF(data_in[idx])[0]
        getFtest =  lambda idx: net.getF(test_data[idx])[0]
        #print F
        #print f
        pred2 = (F.dot(data_in[idx])+f)
        print 'in:',data_in[idx]
        print 'net:',pred
        print 'tay:',pred2
        print 'lbl:',data_out[idx]
        import pdb; pdb.set_trace()


    #Randomize training
    perm = np.random.permutation(data_in.shape[0])
    data_in = data_in[perm]
    data_out = data_out[perm]

    bsize = 50
    lr = 100.0/bsize
    lr_schedule = {
            200000: 0.5,
            400000: 0.5,
            600000: 0.5,
            800000: 0.5,
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
            print 'LR=', lr, ' // Train:',i, net.obj_matrix(data_in, data_out), ' // Test :',i, net.obj_matrix(test_data, test_lbl)
            sys.stdout.flush()
        if i % 10000 == 0:
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
