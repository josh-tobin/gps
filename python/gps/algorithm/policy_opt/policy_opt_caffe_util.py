from caffe import layers as L, NetSpec
from caffe.proto import caffe_pb2

def construct_fc_network(n_layers = 3,
                         Dh = [40,40],
                         Di = 27,
                         Do = 7,
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
        Dh (list): dimensionality of each hidden layer
        Di: dimensionality of input
        Do: dimensionality of the output
        batch_size: batch size
    Returns:
        NetParameter specification of network
    """

    [input, precision, action] = L.MemoryData(ntop=3,
            input_shapes=[dict(dim=[batch_size, Di]),
                          dict(dim=[batch_size,Do,Do]),
                          dict(dim=[batch_size,Do])])
    cur_top = input
    Dh.append(Do)
    for i in range(n_layers):
        cur_top = L.InnerProduct(cur_top,
                                 num_output=Dh[i],
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0))
        # Add nonlinearity to all hidden layers
        if i < n_layers-1:
            cur_top = L.ReLU(cur_top, in_place=True)
    # TODO - the below layer should only exist during training phase
    loss = L.WeightedEuclideanLoss(cur_top, action, precision)
    return loss.to_proto()
