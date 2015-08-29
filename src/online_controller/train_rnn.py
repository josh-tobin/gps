from theano_rnn import *
import theano.tensor as T
import h5py
import logging
import sys

logging.basicConfig(level=logging.DEBUG)

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


def rnntest():
    np.random.seed(123)
    fname = 'net/rnntest.pkl'
    logging.basicConfig(level=logging.DEBUG)

    data, label, clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5', 'data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_lqrtask.hdf5','data/dyndata_armwave_all.hdf5.test'])
    test_data, test_label, test_clip = get_data_hdf5(['data/dyndata_armwave_all.hdf5.train'])
    bsize = 50
    N = data.shape[0]

    try:
        net = unpickle_net(fname)
    except IOError as e:
        ip1 = RNNIPLayer('data', 'ip1', 'clip', 39, 12) 
        act1 = SoftplusLayer('ip1', 'act1')
        ip2 = RNNIPLayer('act1', 'ip2', 'clip', 12, 16) 
        acc = AccelLayer('data', 'ip2', 'acc', 7, 9, 7)
        loss = SquaredLoss('acc', 'lbl')
        net = Network([ip1, act1, ip2, acc], loss)
        #net = unpickle_net('test.pkl')


    net.set_net_inputs([T.matrix('data'), T.matrix('lbl'), T.vector('clip')])
    obj, updates, batch = net.symbolic_forward()
    train_gd = net.get_train_function(obj, updates)
    total_obj = net.get_loss_function(obj, updates)
    rnn_out = net.get_loss_function(batch.get_data('acc'), updates)

    lr = 8e-3/bsize
    lr_schedule = {
        5000000: 0.2,
        10000000: 0.2,
        15000000: 0.2,
    }
    epochs = 0
    for i in range(30*1000*1000):
        bstart = i*bsize % N
        bend = (i+1)*bsize % N
        if bend < bstart:
            epochs += 1
            continue
        _data = data[bstart:bend]
        _label = label[bstart:bend]
        _clip = clip[bstart:bend]
        objval = train_gd(_data, _label, _clip, lr, 0.90)

        if i in lr_schedule:
            lr *= lr_schedule[i]
        if i % 2000 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            sys.stdout.flush()
            #import pdb; pdb.set_trace()
        if i % 10000 == 0:
            if i>0:
                net.pickle(fname)
            total_err = total_obj(data, label, clip)
            print 'Total train error:', total_err

rnntest()
