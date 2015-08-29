from theano_dynamics import *
import numpy as np
import sys
import os
import h5py
import cPickle

DATA_DIR = 'data'
NET_DIR = 'net'

def load_matfile(matfile, remove_ft=False, remove_prevu=False):
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

    if False:
        if 'noprevu' in data:
            print 'Skipping removeprevu!'
        else:
            print 'removing prevu'
            dat = np.c_[dat[:,:14], dat[:,21:]]
            lbl = np.c_[lbl[:,:14], lbl[:,21:]]

    clip = None 
    if 'clip' in data:
        clip = data['clip'][0]
        clip = np.expand_dims(clip, axis=-1)
        print 'Existing CLIP:', clip.shape
    else:
        print 'Auto-generating clip with T=200'
        clip = np.ones((dat.shape[0],1))
        for t in range(dat.shape[0]):
            if t%199 == 0:
                clip[t] = 0
    return dat, lbl, clip

def get_data(train, test, shuffle_test=True, remove_ft=True, remove_prevu=False, clip_dict=None):
    train_data = [] 
    train_label = []
    clip = [] #REMOVE
    for matfile in train:
        #REMOVE
        data_in, data_out, data_clip = load_matfile(os.path.join(DATA_DIR,matfile), remove_ft=remove_ft, remove_prevu=remove_prevu)
        train_data.append(data_in)
        train_label.append(data_out)
        clip.append(data_clip)
        print 'Train data: Loaded %s. Shape: %s' % (matfile, data_in.shape)
    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label)
    if clip_dict is not None:
        clip = np.concatenate(clip) #REMOVE
        clip_dict['clip'] = clip #REMOVE


    test_data = [] 
    test_label = []
    for matfile in test:
        #REMOVE
        data_in, data_out, _ = load_matfile(os.path.join(DATA_DIR,matfile), remove_ft=remove_ft, remove_prevu=remove_prevu)
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
    test2train = 0
    train_data = np.r_[train_data, test_data[:test2train,:]]
    #train_data = np.r_[data_in]
    train_lbl = np.r_[train_label, test_label[:test2train,:]]
    #train_lbl = np.r_[data_out]
    test_data = test_data[test2train:,:]
    test_lbl = test_label[test2train:,:]
    """

    #train_data = train_data[:200,:]
    #train_lbl = train_lbl[:200,:]
    return train_data, train_label, test_data, test_label

def train_dyn():
    fname = os.path.join(NET_DIR, '%s.pkl' % sys.argv[1])
    #np.random.seed(10)
    np.set_printoptions(suppress=True)

    din = 46-7
    dout = 39-7
    data_in, data_out, test_data, test_lbl = get_data(['dyndata_armwave_all'], ['dyndata_plane_nopu'], remove_ft=True, remove_prevu=True)

    #din = 33
    #dout = 26
    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_mjc'], ['dyndata_mjc_test'])

    N = data_in.shape[0]
    print data_in.shape
    print data_out.shape
    print test_data.shape

    # Loss weight
    wt = np.ones(dout).astype(np.float32)

    # Input normalization
    norm1 = NormalizeLayer()
    norm1.generate_weights(data_in)

    """
    #Output normalization
    norm2 = NormalizeLayer()
    norm2.generate_weights(data_out)
    data_out_normed = norm2.forward(data_out)
    norm2.generate_weights(data_out, reverse=True)
    data_out_unnorm = data_out
    data_out = data_out_normed
    """

    #net = NNetDyn([norm1, FFIPLayer(din, 50), ReLU5Layer, FFIPLayer(50,16), AccelLayer()], wt)
    #net = NNetDyn([norm1, FFIPLayer(din, 150), ReLU5Layer, DropoutLayer(150, p=0.8), FFIPLayer(150,dout)], wt)
    #net = NNetDyn([norm1, FFIPLayer(din, 50), SoftPlusLayer, FFIPLayer(50,dout)], wt)
    #net = NNetDyn([norm1, 
    #    GatedLayer([FFIPLayer(din, 50), SigmLayer] ,din, 50),
    #    FFIPLayer(50, dout)],
    #    wt)
    #net = NNetDyn([norm1, FFIPLayer(din, 60), ReLU5Layer, FFIPLayer(60, 60), ReLU5Layer, FFIPLayer(60,dout)], wt)
    net = NNetDyn([FFIPLayer(din,dout)], wt, weight_decay=0.0000) # loss ~0.13

    try:
        net = load_net(fname)
        print 'Loaded net!'
    except IOError:
        print 'Initializing new network!'

    for idx in [25]:
        pred_net =  net.fwd_single(data_in[idx,:din])
        perturbed_input = data_in[idx,:din] + 0.01*np.random.randn(din).astype(np.float32)
        F, f = net.getF(perturbed_input)
        predict_taylor = (F.dot(data_in[idx,:din])+f)
        print 'input:',data_in[idx]
        print 'net_out:',pred_net
        print 'taylor:',predict_taylor
        print 'label:',data_out[idx]
        import pdb; pdb.set_trace()

    #Randomize training
    perm = np.random.permutation(data_in.shape[0])
    data_in = data_in[perm]
    data_out = data_out[perm]

    bsize = 50
    lr = 1e-3/bsize
    lr_schedule = {
            500000: 0.5,
            1000000: 0.5,
            1500000: 0.5,
            2000000: 0.5,
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
        _din = data_in[bstart:bend]
        _dout = data_out[bstart:bend]
        #_din = data_in
        #_dout = data_out

        net.train(_din, _dout, lr, 0.90)

        if i in lr_schedule:
            lr *= lr_schedule[i]

        if i % 2000 == 0:
            print 'LR=', lr, ' // Train:',i, net.obj_matrix(_din, _dout), ' // Test :', net.obj_matrix(test_data, test_lbl)
            sys.stdout.flush()
        if i % 10000 == 0:
            print 'Dumping weights'
            dump_net(fname, net)
            print 'Total train error:', net.obj_matrix(data_in, data_out)

def get_data_hdf5(fnames):
    if type(fnames) == str:
        fnames = [fnames]

    total_data = []
    total_lbl = []
    total_clip = []
    for fname in fnames:
        f = h5py.File(fname)
        total_data.append(f['data'])
        total_lbl.append(f['label'])
        total_clip.append(f['clip'])
    print 'Datasets loaded:', total_data
    print total_lbl
    print total_clip
    return np.concatenate(total_data), \
           np.concatenate(total_lbl), \
           np.concatenate(total_clip)

def merge_datasets():
    train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5', 'data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_all.hdf5.test'])
    import h5py
    f = h5py.File('data/dyndata_plane_combined.hdf5', 'w')
    f['data'] = train_data
    f['label'] = train_lbl
    f['clip'] = train_clip
    pass

def train_dyn_rec():
    np.random.seed(123)
    fname = os.path.join(NET_DIR, '%s.pkl' % sys.argv[1])
    #np.random.seed(10)
    np.set_printoptions(suppress=True)

    dX = 32
    dU = 7
    T = 1
    train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5', 'data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_lqrtask.hdf5','data/dyndata_armwave_all.hdf5.test'])
    #train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train'])
    train_X, train_U, train_tgt = NNetRecursive.prepare_data(train_data, train_lbl, train_clip, dX, dU, T)
    #test_data, test_lbl, test_clip = get_data_hdf5('data/dyndata_plane_expr_nopu2.hdf5')
    #test_X, test_U, test_tgt = NNetRecursive.prepare_data(test_data, test_lbl, test_clip, dX, dU, T)

    # Randomly select a test set out of training data
    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_U = train_U[perm]
    train_tgt = train_tgt[perm]
    Ntrain = int(0.8*train_X.shape[0])
    test_X = train_X[Ntrain:]
    test_U = train_U[Ntrain:]
    test_tgt = train_tgt[Ntrain:]
    train_X = train_X[:Ntrain]
    train_U = train_U[:Ntrain]
    train_tgt = train_tgt[:Ntrain]

    N = train_X.shape[0]
    print 'N:', N, train_X.shape
    print train_U.shape
    print train_tgt.shape



    # Input normalization
    #norm1 = NormalizeLayer()
    #norm1.generate_weights(train_data)

    #net = NNetRecursive(dX, dU, T, [FFIPLayer(dX+dU,dX)], weight_decay=0.0000)
    try:
        with open(fname, 'r') as pklfile:
            layers = cPickle.load(pklfile)
        for layer in layers:
            layer.fix_gpu_vars()
    except:
        print 'Creating new net!'
        # Input normalization
        norm1 = NormalizeLayer()
        norm1.generate_weights(train_data)
        #layers = [norm1, FFIPLayer(dX+dU, 50), ReLULayer, FFIPLayer(50, 50), ReLULayer, FFIPLayer(50, 16+6), AccelLayerFT()] 

        #layers = [norm1, FFIPLayer(dX+dU, 50), ReLULayer, FFIPLayer(50, 16), AccelLayer()] 
        #layers = [FFIPLayer(dX+dU,dX)]

        # Net 3
        #layers = [norm1, FFIPLayer(dX+dU, 80), SoftPlusLayer, DropoutLayer(80, p=0.75), FFIPLayer(80,80), SoftPlusLayer, DropoutLayer(80, p=0.5), FFIPLayer(80, 16), AccelLayer()] 
        layers = [norm1, FFIPLayer(dX+dU, 12), SoftPlusLayer, FFIPLayer(12, 16), AccelLayer()] 

        #layers = [AccelV2Layer([FFIPLayer(dX, 7), SigmLayer],
        #                       [FFIPLayer(dX, 9), SigmLayer])]

        #layers = [ControlAffine([FFIPLayer(dX, 20), SoftPlusLayer, FFIPLayer(20, 16)], 
        #    [FFIPLayer(dX, 16*dU)],
        #    dX, dU, 16), AccelLayer()] 


        # Net 4
        #layers = [
        #    GatedLayer([FFIPLayer(dX+dU, 80), ReLULayer, DropoutLayer(80, p=0.6), FFIPLayer(80,80), SigmLayer] ,dX+dU, 80),
        #    FFIPLayer(80, dX)]

    #"""
    xux = np.c_[train_data, train_lbl]
    mu = np.mean(xux, axis=0)
    diff = xux-mu
    sig = diff.T.dot(diff)
    it = slice(0,dX+dU)
    ip = slice(dX+dU,dX+dU+dX)
    Fm = (np.linalg.pinv(sig[it, it]).dot(sig[it, ip])).T
    fv = mu[ip] - Fm.dot(mu[it]);
    #layers[0].w.set_value(Fm.T)
    #layers[0].b.set_value(fv)
    #"""

    net = NNetRecursive(dX, dU, T, layers, weight_decay=1e-5)

    for i in [25]:
        x = train_X[i]
        u = train_U[i]
        u = u[:dU]
        xu = np.concatenate([x,u])
        tgt = train_tgt[i]
        F, f = net.getF(xu)
        import pdb; pdb.set_trace()

    bsize = 50
    lr = 8.0e-2/bsize
    lr_schedule = {
            400000: 0.2,
            800000: 0.2,
            }

    epochs = 0
    for i in range(1200000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            perm = np.random.permutation(N)
            train_X = train_X[perm]
            train_U = train_U[perm]
            train_tgt = train_tgt[perm]
            continue
        _tX = train_X[bstart:bend]
        _tU = train_U[bstart:bend]
        _tTgt = train_tgt[bstart:bend]
        objval = net.train(_tX, _tU, _tTgt, lr, 0.90)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 1000 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            sys.stdout.flush()
        if i % 10000 == 0:
            if i>0:
                with open(fname, 'w') as pklfile:
                    cPickle.dump(net.layers, pklfile)
            total_err = net.obj(train_X, train_U, train_tgt)
            print 'Total train error:', total_err
            total_err = net.obj(test_X, test_U, test_tgt)
            print 'Total test error:', total_err


def get_net(name, rec=True, dX=32, dU=7):
    if rec:
        net = load_rec_net(name, dX=dX, dU=dU)
    else:
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
    #train_dyn()
    train_dyn_rec()
