from caffe import layers as L, NetSpec
from caffe.proto import caffe_pb2

def construct_fc_network(n_layers = 3,
                         dim_hidden = [40,40],
                         dim_input = 27,
                         dim_output = 7,
                         batch_size = 25):
    """
    Constructs an anonymous network (no layer names) with the specified number
    of inner product layers, and returns NetParameter protobuffer.

    ** Note **: this function is an example for how one might want to specify
    their network, versus providing a protoxt model file. It is not meant to
    be a general solution for specifying any network, as there are many,
    many possible networks one can specify.

    Args:
        n_layers: number of fully connected layers (including output layer)
        dim_hidden (list): dimensionality of each hidden layer
        dim_input: dimensionality of input
        dim_output: dimensionality of the output
        batch_size: batch size
    Returns:
        NetParameter specification of network
    """
    [input, action, precision] = L.DummyData(ntop=3,
            shape=[dict(dim=[batch_size, dim_input]),
                   dict(dim=[batch_size, dim_output]),
                   dict(dim=[batch_size, dim_output, dim_output])])

    #[input, precision, action] = L.MemoryData(ntop=3,
    #        input_shapes=[dict(dim=[batch_size, Di]),
    #                      dict(dim=[batch_size,Do,Do]),
    #                      dict(dim=[batch_size,Do])])
    cur_top = input
    dim_hidden.append(Do)
    for i in range(n_layers):
        cur_top = L.InnerProduct(cur_top,
                                 num_output=dim_hidden[i],
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0))
        # Add nonlinearity to all hidden layers
        if i < n_layers-1:
            cur_top = L.ReLU(cur_top, in_place=True)
    # TODO - the below layer should only exist during training phase
    loss = L.WeightedEuclideanLoss(cur_top, action, precision)
    return loss.to_proto()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

