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

def prep_data():
    #data, label, clip = get_data_hdf5(['data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_lqrtask.hdf5','data/dyndata_armwave_all.hdf5.test'])
    #data, label, clip = get_data_hdf5(['data/dyndata_mjc.hdf5'])
    data, label, clip = get_data_hdf5(['data/dyndata_mjc.hdf5', 'data/dyndata_mjc_expr.hdf5', 'data/dyndata_mjc_expr2.hdf5', 'data/dyndata_mjc_expr3.hdf5'])
    #data, label, clip = get_data_hdf5(['data/dyndata_mjc.hdf5'])

    #data, label, clip = get_data_hdf5(['data/dyndata_workbench_expr.hdf5', 'data/dyndata_workbench.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_gear.hdf5', 'data/dyndata_gear_peg1.hdf5','data/dyndata_gear_peg2.hdf5','data/dyndata_gear_peg3.hdf5','data/dyndata_gear_peg4.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])
    #data, label, clip = get_data_hdf5(['data/dyndata_workbench_expr.hdf5', 'data/dyndata_workbench.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])

    #test_data, test_label, test_clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5'])
    #test_data, test_label, test_clip = get_data_hdf5(['data/dyndata_mjc.hdf5'])

    # Randomly select a test set out of training data
    #fill_clip(clip, k=25) 
    data, label, clip = randomize_dataset(data, label, clip)
    Ntrain = int(0.9*data.shape[0])
    test_data = data[Ntrain:]
    test_label = label[Ntrain:]
    test_clip = clip[Ntrain:]
    data = data[:Ntrain]
    label = label[:Ntrain]
    clip = clip[:Ntrain]
    return data, label, clip, test_data, test_label, test_clip

def train_rnn_step():
    np.random.seed(123)
    #fname = 'net/rnn_plane.pkl'
    fname = sys.argv[1] #'net/gear_rnn.pkl'
    logging.basicConfig(level=logging.DEBUG)

    data, label, clip, test_data, test_label, test_clip = prep_data()
    fill_clip(clip, k=3)
    bsize = 50
    N = data.shape[0]

    djnt = 7
    dee = 6
    dx = 2*dee+2*djnt+0
    du = djnt

    try:
        net = unpickle_net(fname)
        #ffnet = net.to_feedforward_test()
        #ffnet.pickle(fname+'.ff')
    except IOError as e:
        print "Making new net!"
        norm1 = NormalizeLayer('data', 'data_norm')
        norm1.generate_weights(data)

        ip1 = GateV9('data_norm', 'ip1', 'clip', dx+du, 100, activation='relu')
        ip3 = FFIPLayer('ip1', 'ip3', 100, djnt+dee) 
        acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
        loss = SquaredLoss('acc', 'lbl')
        rnet = RecurrentDynamicsNetwork([norm1, ip1,ip3, acc], loss)
        net = rnet.to_feedforward_test()

    net.init_functions(output_blob='acc', weight_decay=1e-5)
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
        F, f, state= net.getF(perturbed_input)
        predict_taylor = (F.dot(data[idx])+f)
        target_label = label[idx]
        import pdb; pdb.set_trace()

    lr = 1e-3/bsize
    lr_schedule = {
        500000: 0.2,
    }
    epochs = 0

    hidden = net.get_init_hidden_state()
    for i in range(1000*1000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            data, label, clip = randomize_dataset(data, label, clip)
            continue
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _clip = clip[bstart:bend]
        net.update(stage=STAGE_TRAIN)

        total_obj = 0
        for j in range(bsize):
            if _clip[j] == 0:
                hidden = net.get_init_hidden_state()
                next_state_action = _data[j]
            objval, next_state, hidden = net.train_gd_step(next_state_action, _label[j], hidden, lr=lr, rho=0.9, momentum=0.99)
            next_state_action = np.r_[next_state, _data[j,dx:dx+du]]
            total_obj += objval

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 200 == 0:
            print 'LR=', lr, ' // Train:',i, total_obj/bsize
            sys.stdout.flush()
            #import pdb; pdb.set_trace()
        if i % 5000 == 0:
            if i>0:
                net.pickle(fname)
                #ffnet = net.to_feedforward_test()
                #ffnet.pickle(fname+'.ff')
            #total_err = net.total_obj(data, label, clip)
            #print 'Total train error:', total_err
            #total_err = net.total_obj(test_data, test_label, test_clip)
            #print 'Total test error:', total_err

def rnntest():
    np.random.seed(123)
    #fname = 'net/rnn_plane.pkl'
    fname = sys.argv[1] #'net/gear_rnn.pkl'
    nettype = int(sys.argv[2]) #'net/gear_rnn.pkl'
    logging.basicConfig(level=logging.DEBUG)

    data, label, clip, test_data, test_label, test_clip = prep_data()
    fill_clip(clip, k=5)
    fill_clip(test_clip, k=5)
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
        """
        if nettype==1:
            ip1 = GRULayer('data_norm', 'ip1', 'clip', dx+du, 80)
            ip2 = GRULayer('ip1', 'ip2', 'clip', 80, 40)
            ip3 = FFIPLayer('ip2', 'ip3', 40, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1, ip2, ip3, acc], loss)
        """
        if nettype==2:  # Try on robot
            ip1 = GateV2('data_norm', 'ip1', 'clip', dx+du, 100, activation='relu')
            ip2 = GateV2('ip1', 'ip2', 'clip', 100, 50, activation='relu')
            ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==4:
            ip1 = GateV6('data_norm', 'ip1', 'clip', dx+du, 100, activation='relu')
            ip2 = GateV6('ip1', 'ip2', 'clip', 100, 50, activation='relu')
            ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==5:  # DOesn't work
                 ip1 = GateV6('data_norm', 'ip1', 'clip', dx+du, 50, activation='relu')
                 ip2 = GateV6('ip1', 'ip2', 'clip', 50, 50, activation='relu')
                 ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee) 
                 acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
                 loss = SquaredLoss('acc', 'lbl')
                 net = RecurrentDynamicsNetwork([norm1, ip1,ip2, ip3, acc], loss)
        if nettype==6: # Bad on robot!
                 ip1 = GateV6('data_norm', 'ip1', 'clip', dx+du, 80, activation='softplus')
                 ip2 = FFIPLayer('ip1', 'ip2', 80, djnt+dee) 
                 acc = AccelLayer('data', 'ip2', 'acc', djnt, dee, du)
                 loss = SquaredLoss('acc', 'lbl')
                 net = RecurrentDynamicsNetwork([norm1, ip1,ip2, acc], loss)
        if nettype==7:
                 ip1 = GateV5('data_norm', 'ip1', 'clip', dx+du, 100, activation='softplus')
                 ip2 = GateV5('ip1', 'ip2', 'clip', 100, 50, activation='softplus')
                 ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee)
                 acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
                 loss = SquaredLoss('acc', 'lbl')
                 net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==8:
                 ip1 = GateV5('data_norm', 'ip1', 'clip', dx+du, 50, activation='softplus')
                 ip2 = GateV5('ip1', 'ip2', 'clip', 50, 50, activation='softplus')
                 ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee)
                 acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
                 loss = SquaredLoss('acc', 'lbl')
                 net = RecurrentDynamicsNetwork([norm1, ip1,ip2, ip3,acc], loss)
        if nettype==9:
                 ip1 = GateV5('data_norm', 'ip1', 'clip', dx+du, 50, activation='softplus')
                 ip2 = GateV5('ip1', 'ip3', 'clip', 50, djnt+dee, activation=None)
                 acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
                 loss = SquaredLoss('acc', 'lbl')
                 net = RecurrentDynamicsNetwork([norm1, ip1,ip2, acc], loss)
        if nettype==10: # Doesn't work
            ip1 = GateV7('data_norm', 'ip1', 'clip', dx+du, 80, activation='softplus')
            ip2 = GateV7('ip1', 'ip2', 'clip', 80, 50, activation='softplus')
            ip3 = FFIPLayer('ip2', 'ip3', 50, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==11:# Doesn't work
            ip1 = GateV7('data_norm', 'ip1', 'clip', dx+du, 50, activation='softplus')
            ip2 = GateV7('ip1', 'ip2', 'clip', 50, 40, activation='softplus')
            ip3 = FFIPLayer('ip2', 'ip3', 40, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip2,ip3, acc], loss)
        if nettype==12: # Works on robot!! --> unstable if I train more
            ip1 = GateV7('data_norm', 'ip1', 'clip', dx+du, 50, activation='softplus')
            ip3 = FFIPLayer('ip1', 'ip3', 50, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip3, acc], loss)
        if nettype==13:
            ip1 = GateV7('data_norm', 'ip1', 'clip', dx+du, 100, activation='softplus')
            ip3 = FFIPLayer('ip1', 'ip3', 100, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip3, acc], loss)
        if nettype==14:  # unstable
            ip1 = GateV8('data_norm', 'ip1', 'clip', dx+du, 100, activation='softplus')
            ip3 = FFIPLayer('ip1', 'ip3', 100, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip3, acc], loss)
        if nettype==15:  # unstable when overtrained
            ip1 = GateV9('data_norm', 'ip1', 'clip', dx+du, 100, activation='relu')
            ip3 = FFIPLayer('ip1', 'ip3', 100, djnt+dee) 
            acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
            loss = SquaredLoss('acc', 'lbl')
            net = RecurrentDynamicsNetwork([norm1, ip1,ip3, acc], loss)
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

    net.init_functions(output_blob='acc', weight_decay=5e-4, train_algo='rmsprop')
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

    lr = 1e-3/bsize
    lr_schedule = {
        200000: 0.2,
    }
    epochs = 0
    for i in range(1000*1000):
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
        objval = net.train_gd(_data, _label, _clip, lr, 0.9, 0.0)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 2000 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            sys.stdout.flush()
            #import pdb; pdb.set_trace()
        if i % 10000 == 0:
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
    #train_rnn_step()
