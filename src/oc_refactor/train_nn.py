import theano.tensor as T
import h5py
import logging
import sys
import scipy.io
import argparse
import os

import common
from dynamics_nn import *

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)
np.random.seed(123)

LOGGER = logging.getLogger(__name__)

def load_hdf5(fnames):
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

def load_mat(fname):
    if type(fnames) == str:
        fnames = [fnames]

    total_data = []
    total_lbl = []
    total_clip = []
    for fname in fnames:
        f = scipy.io.loadmat(fname)
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


def prep_data_prevsa(data_files = None):
    data, label, clip = load_mat(data_files)

    N = data.shape[0]
    prevsa = np.zeros_like(data)
    for n in range(N):
        if clip[n] == 0:
            prevsa[n,:] = data[n]
            continue
        prevsa[n,:] = data[n-1]  #data[n-1,:]
    data, label, prevsa = randomize_data(N, data, label, prevsa)
    return data, label, prevsa

def linearize_debug(net, xu, prevxu):
    """Printout linearization of network around a datapoint
    Linearization should roughly follow the block structure:
    [I tI t^2I]
    [0 I  tI]
    Which denotes the physics equations
        x = x + t*v + t^2*a
        v = v + t*a
    """
    perturbed_input = xu
    F, f = net.linearize(perturbed_input, prevxu)
    predict_taylor = (F.dot(xu)+f)
    print 'Linearization:'
    print '\tF:', F
    print '\tf:', f

def train_nn(fname, new=True, data_files=None, update_every=2000, save_every=10000):
    logging.basicConfig(level=logging.DEBUG)

    data, label, prevsa = prep_data_prevsa(data_files)
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
            # Shuffle data after iterating through epoch
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
            linearize_debug(net, data[0], prevsa[0])

def parse_args():
    parser = argparse.ArgumentParser(description='Train/run online controller in MuJoCo')
    parser.add_argument('-f', '--filename', type=str, help='Network filename')
    parser.add_argument('-n', '--new', action='store_true', default=False, help='Create a new network')

    default_data = [os.path.join('data', common.OFFLINE_DYNAMICS_DATA)]
    parser.add_argument('-d', '--data', type=str, metavar='N', nargs='+', default=default_data, help='Data files')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train_nn(args.filename, new=args.new, data_files=args.data)
