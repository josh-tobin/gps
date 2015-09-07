from theano_dynamics import *
import numpy as np
import sys
import os
import h5py
import cPickle

DATA_DIR = 'data'
NET_DIR = 'net'

def get_net(name, rec=True, dX=32, dU=7):
    if rec:
        net = load_rec_net(name, dX=dX, dU=dU)
    else:
        net = load_net(name)
    return net

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
        clip = data['clip'].T.astype(np.float32)#[0]
        #clip = np.expand_dims(clip, axis=-1)
        print 'Existing CLIP:', clip.shape
    else:
        print 'Auto-generating clip with T=100'
        clip = np.ones((dat.shape[0],1))
        for t in range(dat.shape[0]):
            if t%99 == 0:
                clip[t] = 0
    return dat, lbl, clip

def get_data(train, test, shuffle_test=True, remove_ft=True, remove_prevu=False, clip_dict=None):
    """ Get data in matfile format """
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
    #np.random.seed(123)
    fname = os.path.join(NET_DIR, '%s.pkl' % sys.argv[1])
    #np.random.seed(10)
    np.set_printoptions(suppress=True)

    dX = 32#26
    dU = 7
    T = 1
    # Gear datasets: dyndata_gear, dyndata_gear_peg1, dyndata_gear_peg2, dyndata_gear_peg3, dyndata_gear_peg4,  dyndata_armwave_lqrtask, dyndata_armwave_all.train,
    #train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5', 'data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_lqrtask.hdf5','data/dyndata_armwave_all.hdf5.test'])
    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_armwave_all'], ['dyndata_plane_nopu'], remove_ft=True, remove_prevu=True)
    #train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_mjc_expr.hdf5', 'data/dyndata_mjc_expr2.hdf5'])
    #train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_workbench_expr.hdf5', 'data/dyndata_workbench.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_gear.hdf5', 'data/dyndata_gear_peg1.hdf5','data/dyndata_gear_peg2.hdf5','data/dyndata_gear_peg3.hdf5','data/dyndata_gear_peg4.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])
    train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_block.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_gear.hdf5', 'data/dyndata_gear_peg1.hdf5','data/dyndata_gear_peg2.hdf5','data/dyndata_gear_peg3.hdf5','data/dyndata_gear_peg4.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])
    #train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_workbench_expr.hdf5', 'data/dyndata_workbench.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])

    #train_data, train_lbl, train_clip = get_data_hdf5(['data/dyndata_car.hdf5', 'data/dyndata_gear_peg4.hdf5', 'data/dyndata_armwave_lqrtask.hdf5'])
    train_X, train_U, train_tgt = NNetRecursive.prepare_data(train_data, train_lbl, train_clip, dX, dU, T)
    #test_data, test_lbl, test_clip = get_data_hdf5('data/dyndata_plane_expr_nopu2.hdf5')
    #test_X, test_U, test_tgt = NNetRecursive.prepare_data(test_data, test_lbl, test_clip, dX, dU, T)

    # Randomly select a test set out of training data
    perm = np.random.permutation(train_X.shape[0])
    train_X = train_X[perm]
    train_U = train_U[perm]
    train_tgt = train_tgt[perm]
    Ntrain = int(0.9*train_X.shape[0])
    test_X = train_X[Ntrain:]
    test_U = train_U[Ntrain:]
    test_tgt = train_tgt[Ntrain:]
    train_X = train_X[:Ntrain]
    train_U = train_U[:Ntrain]
    train_tgt = train_tgt[:Ntrain]

    N = train_X.shape[0]
    print 'N train:', N, train_X.shape
    print 'U shape:', train_U.shape

    try:
        with open(fname, 'r') as pklfile:
            layers = cPickle.load(pklfile)
    except IOError as e:
        print 'Creating new net!'
        # Input normalization
        norm1 = NormalizeLayer()
        norm1.generate_weights(train_data)

        # Least Squares
        #layers = [FFIPLayer(dX+dU,dX)]

        # Acceleration Net
        #layers = [norm1, FFIPLayer(dX+dU, 80), SoftPlusLayer, DropoutLayer(80, p=0.75), FFIPLayer(80,80), SoftPlusLayer, DropoutLayer(80, p=0.5), FFIPLayer(80, 16), AccelLayer()] 
        #layers = [norm1, FFIPLayer(dX+dU, 200), DropoutLayer(200, p=0.5), ReLULayer, FFIPLayer(200,100), DropoutLayer(100, p=0.5), ReLULayer, FFIPLayer(100, 16), AccelLayer()] 
        #layers = [norm1, FFIPLayer(dX+dU, 50), SoftPlusLayer, FFIPLayer(50, 16), AccelLayer()] 
        #layers = [norm1, FFIPLayer(dX+dU, 100), SoftPlusLayer, FFIPLayer(100, 50), SoftPlusLayer, FFIPLayer(50, 16) , AccelLayer()] 
        #layers = [norm1, FFIPLayer(dX+dU, 50), SoftPlusLayer, FFIPLayer(50, 30), SoftPlusLayer, FFIPLayer(30, 16) , AccelLayer()] 
        layers = [norm1, FFIPLayer(dX+dU, 100), ReLULayer, FFIPLayer(100, 16) , AccelLayer()] 

        # Gated net
        #layers = [
        #    GatedLayer([FFIPLayer(dX+dU, 80), ReLULayer, DropoutLayer(80, p=0.6), FFIPLayer(80,80), SigmLayer] ,dX+dU, 80),
        #    FFIPLayer(80, dX)]

    """
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
    """

    net = NNetRecursive(dX, dU, T, layers, weight_decay=1e-4)

    for i in [25]:
        x = train_X[i]
        u = train_U[i]
        u = u[:dU]
        xu = np.concatenate([x,u])
        tgt = train_tgt[i]
        F, f = net.getF(xu)
        import pdb; pdb.set_trace()

    bsize = 50
    lr = 8e-2/bsize
    lr_schedule = {
            100000: 0.2,
            200000: 0.2,
            400000: 0.2,
            }

    epochs = 0
    for i in range(5200000):
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
        if i % 2000 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            sys.stdout.flush()
        if i % 20000 == 0:
            if i>0:
                #net.make_sigma_x(train_X, train_U, train_tgt)
                with open(fname, 'w') as pklfile:
                    cPickle.dump(net.layers, pklfile)
            total_err = net.obj(train_X, train_U, train_tgt)
            print 'Total train error:', total_err
            total_err = net.obj(test_X, test_U, test_tgt)
            print 'Total test error:', total_err



if __name__ == "__main__":
    #get_net()
    #train_dyn()
    train_dyn_rec()
