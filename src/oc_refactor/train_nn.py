from dynamics_nn import *
import theano.tensor as T
import h5py
import logging
import sys
import scipy.io
import argparse

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)
np.random.seed(123)

LOGGER = logging.getLogger(__name__)

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
    LOGGER.info('Datasets loaded:')
    LOGGER.info('>',total_data)
    LOGGER.info('>',total_lbl)
    LOGGER.info('>',total_clip)
    return np.concatenate(total_data).astype(np.float32), \
           np.concatenate(total_lbl).astype(np.float32), \
           np.concatenate(total_clip).astype(np.float32)[:,0]

def randomize_data(N, *matrices):
    randomized = []
    perm = np.random.permutation(N)
    for mat in matrices:
        randomized.append(mat[perm])
    return randomized


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
    data, label, prevsa = randomize_data(N, data, label, prevsa)
    return data, label, prevsa

def train_nn(fname, new=True, update_every=2000, save_every=10000):
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
        if new:
            print "Making new network!"
        else:
            raise e

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


    net.init_functions(output_blob='acc', weight_decay=1e-4, train_algo='rmsprop')
    """
    # Check jacobian of network around a data point
    for idx in [5]:
        perturbed_input = data[idx] 
        F, f = net.linearize(perturbed_input, prevsa[idx])
        predict_taylor = (F.dot(data[idx])+f)
        target_label = label[idx]
        import pdb; pdb.set_trace()
    """

    lr = 5e-3/bsize
    lr_schedule = {
        80000: 0.2,
        200000: 0.2,
        1000000: 0.2,
        2000000: 0.2,
    }
    epochs = 0

    # TRAINING LOOP
    for i in range(3*1000*1000):

        # Compute batch
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            data, label, prevsa = randomize_data(N, data, label, prevsa)
            continue
        # Select a minibatch
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _prevxu = prevsa[bstart:bend]

        # Compute a GD step
        net.update(stage=STAGE_TRAIN)
        objval = net.train_gd(_data, _prevxu, _label, lr, 0.9, 0)

        # LR decay/messages
        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % update_every == 0:
            print 'Epoch=%d, Iter=%d, LR=%f, Objective=%f' % (epoch, i, lr, objval)
            sys.stdout.flush()
        if i % save_every == 0:
            if i>0:
                net.pickle(fname)
            total_err = net.total_obj(data, prevsa, label)
            print 'Saving network. Total train error:', total_err

def parse_args():
    parser = argparse.ArgumentParser(description='Train/run online controller in MuJoCo')
    parser.add_argument('-f', '--filename', type=str, help='Network filename')
    parser.add_argument('-n', '--new', action='store_true', default=False, help='Create a new network')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train_nn(args.filename, new=args.new)
