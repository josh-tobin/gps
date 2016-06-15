from dynamics_nn import *
import theano.tensor as T
import h5py
import logging
import sys
import scipy.io

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)
np.random.seed(123)

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
    #data, label, clip = get_data_hdf5(['data/dyndata_workbench_expr.hdf5', 'data/dyndata_workbench.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])
    data, label, clip = get_data_hdf5(['data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_gear.hdf5', 'data/dyndata_gear_peg1.hdf5','data/dyndata_gear_peg2.hdf5','data/dyndata_gear_peg3.hdf5','data/dyndata_gear_peg4.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])
    #data, label, clip = get_data_hdf5(['data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])

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

def prep_data_prevsa():
    #data, label, clip = get_data_hdf5(['data/dyndata_workbench_expr.hdf5', 'data/dyndata_workbench.hdf5', 'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'])
    #"""
    data, label, clip = get_data_hdf5([
        'data/dyndata_plane_expr_nopu.hdf5', 'data/dyndata_plane_expr_nopu2.hdf5', 'data/dyndata_plane_nopu.hdf5',
        'data/dyndata_workbench_expr.hdf5', 
        'data/dyndata_workbench.hdf5',
        'data/dyndata_reverse_ring.hdf5', 'data/dyndata_plane_table.hdf5', 'data/dyndata_plane_table_expr.hdf5', 'data/dyndata_car.hdf5', 
        'data/dyndata_gear.hdf5', 
        'data/dyndata_gear_peg1.hdf5','data/dyndata_gear_peg2.hdf5','data/dyndata_gear_peg3.hdf5','data/dyndata_gear_peg4.hdf5', 
        'data/dyndata_armwave_lqrtask.hdf5', 'data/dyndata_armwave_all.hdf5.train', 'data/dyndata_armwave_still.hdf5'
        ])
    #"""
    #data, label, clip = get_data_hdf5(['data/dyndata_mjc_expr.hdf5', 'data/dyndata_mjc_expr2.hdf5', 'data/dyndata_mjc_expr3.hdf5'])
    N = data.shape[0]
    prevsa = np.zeros_like(data)
    for n in range(N):
        if clip[n] == 0:
            prevsa[n,:] = data[n]
            continue
        prevsa[n,:] = data[n-1]  #data[n-1,:]
    perm = np.random.permutation(N)
    data = data[perm]
    label = label[perm]
    prevsa = prevsa[perm]
    return data, label, prevsa

def train_nn(fname):
    logging.basicConfig(level=logging.DEBUG)

    data, label, prevsa = prep_data_prevsa()
    bsize = 50
    N = data.shape[0]

    djnt = 7
    dee = 6
    dx = 2*dee+2*djnt+0
    du = djnt

    try:
        net = unpickle_net(fname)
    except IOError as e:
        print "Making new net!"

        #sub1 = SubtractAction('data', 'prevxu', 'data_sub')

        norm1 = NormalizeLayer('data', 'data_norm')
        norm1.generate_weights(data)

        norm2 = NormalizeLayer('prevxu', 'prevxu_norm')
        norm2.generate_weights(prevsa)

        ip1 = PrevSALayer('data_norm', 'prevxu_norm', 'ip1', dx+du, 100, dx+du)
        #ip1 = PrevSALayer2('data', 'prevxu', 'ip1', dx, du, 80)
        act1 = ReLULayer('ip1', 'act1')
        ip2 = FFIPLayer('act1', 'ip2', 100, 30) 
        act2 = ReLULayer('ip2', 'act2')
        ip3 = FFIPLayer('act2', 'ip3', 30, djnt+dee) 
        acc = AccelLayer('data', 'ip3', 'acc', djnt, dee, du)
        loss = SquaredLoss('acc', 'lbl')
        net = PrevSADynamicsNetwork([norm1, norm2, ip1, act1, ip2, act2, ip3, acc], loss)


    losswt = np.ones(dx)
    losswt[0:7] = 1.0
    net.loss.wt = losswt

    net.init_functions(output_blob='acc', weight_decay=1e-4, train_algo='rmsprop')
    for idx in [5]:
        #pred_net =  net.fwd_single(data[idx])
        perturbed_input = data[idx] 
        #+ 0.01*np.random.randn(dx+du).astype(np.float32)
        F, f = net.getF(perturbed_input, prevsa[idx])
        predict_taylor = (F.dot(data[idx])+f)
        target_label = label[idx]
        import pdb; pdb.set_trace()

    lr = 5e-3/bsize
    lr_schedule = {
        80000: 0.2,
        200000: 0.2,
        1000000: 0.2,
        2000000: 0.2,
    }
    epochs = 0
    for i in range(3*1000*1000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            perm = np.random.permutation(N)
            data = data[perm]
            label = label[perm]
            prevsa = prevsa[perm]
            #data, label, clip = randomize_dataset(data, label, clip)
            continue
        net.clear_recurrent_state()
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _prevxu = prevsa[bstart:bend]
        net.update(stage=STAGE_TRAIN)
        objval = net.train_gd(_data,_prevxu, _label, lr, 0.9, 0)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 2000 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            sys.stdout.flush()
            #import pdb; pdb.set_trace()
        if i % 10000 == 0:
            if i>0:
                #net.calculate_sigmax(data, prevsa, label)
                net.pickle(fname)
            total_err = net.total_obj(data, prevsa, label)
            print 'Total train error:', total_err

if __name__ == "__main__":
    train_nn(sys.argv[1])
