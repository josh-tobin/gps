import caffe
import argparse
import numpy as np

import theano_dynamics

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('caffemodel', type=str, help='Caffe model to load')
    parser.add_argument('caffenet', type=str, help='Caffe net definition prototxt')
    parser.add_argument('outfile', type=str, help='File to dump theano net')
    args = parser.parse_args()
    return args

def get_relu(layer):
    return theano_dynamics.ReLULayer

def get_ip(layer):
    wt = layer.blobs[0].data
    bias = layer.blobs[1].data
    print '> Shape: %s + %s' % (str(wt.shape), str(bias.shape))
    iplayer = theano_dynamics.FFIPLayer(wt.shape[1], wt.shape[0])
    iplayer.set_weights(wt.T, bias)
    return iplayer

LAYER_DICT = {
    "InnerProduct": get_ip,
    "ReLU": get_relu
}

def main():
    args = parse_args()
    net = caffe.Net(args.caffenet, args.caffemodel, caffe.TRAIN)
    layers = net.layers
    theano_layers = []
    for layer in layers:
        if layer.type in LAYER_DICT:
            print 'Loading %s layer' % layer.type
            theano_layers.append(LAYER_DICT[layer.type](layer))
        else:
            print 'Unknown layer %s; Skipping' % layer.type

    theano_net = theano_dynamics.NNetDyn(theano_layers, np.ones(39))
    theano_dynamics.dump_net(args.outfile, theano_net)

if __name__ == "__main__":
    main()
