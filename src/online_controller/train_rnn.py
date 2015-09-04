from theano_rnn import *
import theano.tensor as T
import h5py
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)

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
    print 'Datasets loaded:'
    print '>',total_data
    print '>',total_lbl
    print '>',total_clip
    return np.concatenate(total_data).astype(np.float32), \
           np.concatenate(total_lbl).astype(np.float32), \
           np.concatenate(total_clip).astype(np.float32)[:,0]

def fill_clip(existing_clip, k=2):
    existing_clip.fill(1.0)
    for i in range(existing_clip.shape[0]):
        if i%k == 0 and existing_clip[i]==1.0:
            existing_clip[i] = 0.0

def randomize_dataset(data, label, clip):
    """
    >>> np.random.seed(12345)
    >>> data = np.array(np.arange(10))
    >>> label = np.array(np.arange(10))+1
    >>> clip = np.ones(10)
    >>> fill_clip(clip, k=2)
    >>> new_data, new_label, new_clip = randomize_dataset(data, label, clip)
    >>> new_data
    array([0, 1, 8, 9, 6, 7, 2, 3, 4, 5])
    >>> new_label
    array([ 1,  2,  9, 10,  7,  8,  3,  4,  5,  6])
    >>> new_clip
    array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.])
    """
    N = data.shape[0]
    boundary_idx = np.array(np.nonzero(1-clip)).T
    perm = np.random.permutation(len(boundary_idx))
    boundary_idx = boundary_idx[perm]

    output_data = np.zeros_like(data)
    output_label = np.zeros_like(label)
    output_clip = np.zeros_like(clip)

    out_idx = 0
    for bidx in boundary_idx:
        assert(clip[bidx] == 0)
        idx = bidx
        cur_clip = 1.0
        while cur_clip == 1.0:
            output_data[out_idx] = data[idx]
            output_label[out_idx] = label[idx]
            output_clip[out_idx] = clip[idx]
            out_idx += 1
            idx += 1
            if idx >= N:
                break
            cur_clip = clip[idx]
    return output_data, output_label, output_clip


def rnntest():
    np.random.seed(123)
    #fname = 'net/rnn_plane.pkl'
    fname = sys.argv[1] #'net/gear_rnn.pkl'
    nettype = int(sys.argv[2]) #'net/gear_rnn.pkl'
    logging.basicConfig(level=logging.DEBUG)

    #data, label, clip = get_data_hdf5(['data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_lqrtask.hdf5','data/dyndata_armwave_all.hdf5.test'])
    #data, label, clip = get_data_hdf5(['data/dyndata_mjc.hdf5'])
    data, label, clip = get_data_hdf5(['data/dyndata_mjc.hdf5', 'data/dyndata_mjc_expr.hdf5', 'data/dyndata_mjc_expr2.hdf5'])
    #data, label, clip = get_data_hdf5(['data/dyndata_mjc.hdf5'])
    #data, label, clip = get_data_hdf5(['data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_gear.hdf5', 'data/dyndata_gear_peg1.hdf5','data/dyndata_gear_peg2.hdf5','data/dyndata_gear_peg3.hdf5','data/dyndata_gear_peg4.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train'])

    #test_data, test_label, test_clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5'])
    #test_data, test_label, test_clip = get_data_hdf5(['data/dyndata_mjc.hdf5'])

    # Randomly select a test set out of training data
    fill_clip(clip, k=10) 
    data, label, clip = randomize_dataset(data, label, clip)
    Ntrain = int(0.9*data.shape[0])
    test_data = data[Ntrain:]
    test_label = label[Ntrain:]
    test_clip = clip[Ntrain:]
    data = data[:Ntrain]
    label = label[:Ntrain]
    clip = clip[:Ntrain]

    bsize = 50
    N = data.shape[0]

    djnt = 7
    dee = 6
    dx = 2*dee+2*djnt+0
    du = djnt

    try:
        net = unpickle_net(fname)
        ffnet = net.to_feedforward_test()
        ffnet.pickle(fname+'.ff')
    except IOError as e:
        print "Making new net!"
        norm1 = NormalizeLayer('data', 'data_norm')
        norm1.generate_weights(data)

        #"""
        if nettype==1:
            #ip1 = RNNIPLayer('data', 'ip1', 'clip', dx+du, 100, activation='tanh') 
            ip1 = GRULayer('data_norm', 'ip1', 'clip', dx+du, 100)
            drop1 = DropoutLayer('ip1', 'drop1', 100)
            ip2 = GRULayer('drop1', 'ip2', 'clip', 100, 50)
            ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1, drop1, ip2, ip3, acc], loss)
            #"""
        if nettype==2:
            ip1 = SimpGateLayer('data_norm', 'ip1', 'clip', dx+du, 50)
            ip2 = SimpGateLayer('ip1', 'ip2', 'clip', 50, 30)
            ip3 = FFIPLayer('ip2', 'ip3', 30, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==3:
            ip1 = GRULayer('data_norm', 'ip1', 'clip', dx+du, 80)
            ip2 = GRULayer('drop1', 'ip2', 'clip', 80, 30)
            ip3 = FFIPLayer('ip2', 'ip3', 30, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==4:
            ip1 = RNNIPLayer('data_norm', 'ip1', 'clip', dx+du, 100, activation='tanh')
            ip2 = RNNIPLayer('ip1', 'ip2', 'clip', 100, 30, activation='tanh')
            ip3 = FFIPLayer('ip2', 'ip3', 30, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)

        """
        ip1 = RNNIPLayer('data_norm', 'ip1', 'clip', dx+du, 40, activation='tanh')
        ip2 = FFIPLayer('ip1', 'ip2', 40, djnt+dee)
        acc = AccelLayer('data', 'ip2', 'acc', djnt, dee, du)
        loss = SquaredLoss('acc', 'lbl')
        net = RecurrentDynamicsNetwork([norm1, ip1, ip2, acc], loss)
        """

        """
        ip1 = FFIPLayer('data', 'ip1', dx+du, 12) 
        act1 = SoftplusLayer('ip1', 'act1')
        ip2 = FFIPLayer('act1', 'ip2', 12, djnt+dee) 
        acc = AccelLayer('data', 'ip2', 'acc', djnt, dee, du)
        loss = SquaredLoss('acc', 'lbl')
        net = RecurrentDynamicsNetwork([ip1, act1, ip2, acc], loss)
        """

    net.init_functions(output_blob='acc', weight_decay=1e-5, train_algo='rmsprop')
    """
    net.set_net_inputs([T.matrix('data'), T.matrix('lbl'), T.vector('clip')])
    obj, updates = net.symbolic_forward()
    train_gd = net.get_train_function(obj, updates)
    total_obj = net.get_loss_function(obj, updates)
    rnn_out = net.get_loss_function(batch.get_data('acc'), updates)
    """
    for idx in [0]:
        #pred_net =  net.fwd_single(data[idx])
        perturbed_input = data[idx] + 0.01*np.random.randn(dx+du).astype(np.float32)
        F, f = net.getF(perturbed_input)
        predict_taylor = (F.dot(data[idx])+f)
        target_label = label[idx]
        #import pdb; pdb.set_trace()

    lr = 5e-2/bsize
    lr_schedule = {
        1000000: 0.2,
        2000000: 0.2,
    }
    epochs = 0
    for i in range(3*1000*1000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            data, label, clip = randomize_dataset(data, label, clip)
            continue
        net.clear_recurrent_state()
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _clip = clip[bstart:bend]
        net.update(stage=STAGE_TRAIN)
        objval = net.train_gd(_data, _label, _clip, lr, 0.9, 0.9)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 2000 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            sys.stdout.flush()
            #import pdb; pdb.set_trace()
        if i % 30000 == 0:
            if i>0:
                net.pickle(fname)
                ffnet = net.to_feedforward_test()
                ffnet.pickle(fname+'.ff')
            #total_err = net.total_obj(data, label, clip)
            #print 'Total train error:', total_err
            total_err = net.total_obj(test_data, test_label, test_clip)
            print 'Total test error:', total_err

if __name__ == "__main__":
    rnntest()
