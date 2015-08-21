from theano_dynamics import *
import numpy as np
import sys
import os

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

    if remove_prevu:
        if 'noprevu' in data:
            print 'Skipping removeprevu!'
        else:
            dat = np.c_[dat[:,:14], dat[:,21:]]
            lbl = np.c_[lbl[:,:14], lbl[:,21:]]
    return dat, lbl
        

def get_data(train, test, shuffle_test=True, remove_ft=True, remove_prevu=False):
    train_data = [] 
    train_label = []
    for matfile in train:
        data_in, data_out = load_matfile(os.path.join(DATA_DIR,matfile), remove_ft=remove_ft, remove_prevu=remove_prevu)
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
        data_in, data_out = load_matfile(os.path.join(DATA_DIR,matfile), remove_ft=remove_ft, remove_prevu=remove_prevu)
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

    #din = 46-7
    #dout = 39-7
    #data_in, data_out, test_data, test_lbl = get_data(['dyndata_plane_nopu', 'dyndata_armwave', 'dyndata_armwave_2', 'dyndata_armwave_moretq', 'dyndata_armwave_moretq2', 'dyndata_armwave_moretq', 'dyndata_armwave_moretq3'], ['dyndata_plane_nopu'], remove_ft=True)

    din = 33
    dout = 26
    data_in, data_out, test_data, test_lbl = get_data(['dyndata_mjc'], ['dyndata_mjc_test'])

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
    net = NNetDyn([norm1, 
        GatedLayer([FFIPLayer(din, 50), SigmLayer] ,din, 50),
        FFIPLayer(50, dout)],
        wt)
    #net = NNetDyn([norm1, FFIPLayer(din, 60), ReLU5Layer, FFIPLayer(60, 60), ReLU5Layer, FFIPLayer(60,dout)], wt)
    #net = NNetDyn([FFIPLayer(din,dout)], wt, weight_decay=0.0000) # loss ~0.13

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
    lr = 5e-3/bsize
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
