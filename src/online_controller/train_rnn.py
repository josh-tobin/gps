from theano_rnn import *
import theano.tensor as T
import h5py

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
    logging.basicConfig(level=logging.DEBUG)

    data, label, clip = get_data_hdf5(['data/dyndata_plane_expr_nopu2.hdf5', 'data/dyndata_plane_nopu.hdf5','data/dyndata_plane_expr_nopu.hdf5','data/dyndata_armwave_lqrtask.hdf5','data/dyndata_armwave_all.hdf5.test'])
    bsize = 50
    N = data.shape[0]

    ip1 = RNNIPLayer('data', 'ip1', 'clip', 39, 32) 
    loss = SquaredLoss('ip1', 'lbl')
    net = Network([ip1], loss)
    #net = unpickle_net('test.pkl')

    net.set_net_inputs([T.matrix('data'), T.matrix('lbl'), T.vector('clip')])
    obj = net.symbolic_forward()
    train_gd = net.get_train_function(obj)
    total_obj = net.get_loss_function(obj)

    lr = 5e-3/bsize
    lr_schedule = {
        400000: 0.2,
        800000: 0.2,
    }
    epochs = 0
    for i in range(100000):
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
        if i % 500 == 0:
            print 'LR=', lr, ' // Train:',i, objval
            #import pdb; pdb.set_trace()
        if i % 10000 == 0:
            pass
            #if i>0:
            #    net.pickle('test.pkl')
            total_err = total_obj(data, label, clip)
            print 'Total train error:', total_err

rnntest()