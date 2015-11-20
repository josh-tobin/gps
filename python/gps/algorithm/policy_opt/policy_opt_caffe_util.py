from caffe import layers as L, NetSpec
from caffe.proto.caffe_pb2 import TRAIN, TEST
import json

def construct_fc_network(n_layers = 3,
                         dim_hidden = [40,40],
                         dim_input = 27,
                         dim_output = 7,
                         batch_size = 25,
                         phase = TRAIN):
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
    data_layer_info = json.dumps({
            'shape': [{'dim': (batch_size, dim_input)},
                      {'dim': (batch_size, dim_output)},
                      {'dim': (batch_size, dim_output, dim_output)}]})

    if phase == TRAIN:
        [input, action, precision] = L.Python(ntop=3,
                                                   python_param=dict(
                                                   module='policy_layers',
                                                   param_str = data_layer_info,
                                                   layer = 'PolicyDataLayer'))
        #[input, action, precision] = L.DummyData(ntop=3,
        #        shape=[dict(dim=[batch_size, dim_input]),
        #        dict(dim=[batch_size, dim_output]),
        #        dict(dim=[batch_size, dim_output, dim_output])])
    else:
        [input, action, precision] = L.Python(ntop=3,
                                                   python_param=dict(
                                                   module='policy_layers',
                                                   param_str = data_layer_info,
                                                   layer = 'PolicyDataLayer'))
        input = L.DummyData(ntop=1,
                shape=[dict(dim=[batch_size, dim_input])])

    cur_top = input
    dim_hidden.append(dim_output)
    for i in range(n_layers):
        cur_top = L.InnerProduct(cur_top,
                                 num_output=dim_hidden[i],
                                 weight_filler=dict(type='gaussian', std=0.01),
                                 bias_filler=dict(type='constant', value=0))
        # Add nonlinearity to all hidden layers
        if i < n_layers-1:
            cur_top = L.ReLU(cur_top, in_place=True)

    if phase == TRAIN:
        out = L.WeightedEuclideanLoss(cur_top, action, precision)
    else:
        out = cur_top

    return out.to_proto()
