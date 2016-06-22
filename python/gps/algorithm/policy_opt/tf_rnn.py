import tensorflow as tf
from tensorflow.python.framework import ops
from gps.algorithm.policy_opt.tf_utils import TfMap
from gps.algorithm.policy_opt.tf_model_example import get_loss_layer, \
                                            batched_matrix_vector_multiply
import numpy as np

def time_distributed_euclidean_loss_layer2(a, b, precision, output_shape):
    u = tf.reshape(a-b, [-1, output_shape])
    uP = tf.matmul(u, precision)
    uPu = tf.matmul(uP, u)
    loss = tf.reduce_mean(tf.diag_part(uPu))
    return loss
def time_distributed_euclidean_loss_layer(a, b, precision):
    ''' return the average of uPu, i.e., (a-b)'*precision*(a-b), avgd over
        timesteps and batches '''
    # first, make u (batch_size, n_steps, 1, dim_output)-shaped
    u_left_expand = tf.expand_dims(a-b, -2)
    uP = tf.batch_matmul(u_left_expand, precision)
    # next, we want u to be (batch_size, n_steps, dim_output, 1)-shaped
    u_right_expand = tf.expand_dims(a-b, -1)
    uPu = tf.reduce_mean(tf.batch_matmul(uP, u_right_expand))
    return uPu

def time_distributed_multi_input_dense_layer(x1, x2, x1_dim, x2_dim,
                                             output_dim, activation=tf.nn.relu,
                                             name=None):
    x_dim = tf.shape(x1[:,:,0]) # assume the first 2 dims are the same
    target_shape = tf.concat(0, [x_dim, (output_dim,)])
    n = x1_dim + x2_dim
    init_W1 = np.random.randn(x1_dim, output_dim)*np.sqrt(2.0/n)
    init_W2 = np.random.randn(x2_dim, output_dim)*np.sqrt(2.0/n)
    init_b = np.zeros([output_dim,])
    init_W1 = init_W1.astype(np.float32)
    init_W2 = init_W2.astype(np.float32)
    init_b = init_b.astype(np.float32)
    W1 = tf.Variable(init_W1)
    W2 = tf.Variable(init_W2)
    b = tf.Variable(init_b)

    with ops.op_scope([x1, x2, W1, W2, b], name, 'td_mi_layer') as name:
        x1t = ops.convert_to_tensor(x1, name='x1')
        x2t = ops.convert_to_tensor(x2, name='x2')
        W1t = ops.convert_to_tensor(W1, name='W1')
        W2t = ops.convert_to_tensor(W2, name='W2')
        bt = ops.convert_to_tensor(b, name='b')

        x1W1 = tf.matmul(tf.reshape(x1t, [-1, x1_dim]), W1t)
        x2W2 = tf.matmul(tf.reshape(x2t, [-1, x2_dim]), W2t)
        xW = x1W1 + x2W2
        xW_plus_b = tf.nn.bias_add(xW, bt)
        xW_plus_b = tf.reshape(xW_plus_b, target_shape)
        if activation == None or activation == 'linear':
            out = xW_plus_b
        else:
            out = activation(xW_plus_b)
        return out, [W1, W2], [b]

def time_distributed_dense_layer(x, input_dim, output_dim, 
                                 activation=tf.nn.relu, 
                                 name=None):
    ''' Apply the same dense layer for each time timension of an
        input sequence. Used after a recurrent layer.'''
    print "BUILDING TDDL with input_dim %d, output_dim %d"%(input_dim, output_dim)
    initial_weights = (np.random.randn(input_dim, output_dim)
                       *np.sqrt(2.0/input_dim)).astype(np.float32)
    initial_bias = np.zeros([output_dim,]).astype(np.float32)
    W = tf.Variable(initial_weights)
    b = tf.Variable(initial_bias)
    
    with ops.op_scope([x, W, b], name, 'td_layer') as name:
        xt = ops.convert_to_tensor(x, name='x')
        Wt = ops.convert_to_tensor(W, name='W')
        bt = ops.convert_to_tensor(b, name='b')
        original_shape = tf.shape(xt[:,:,0])
        new_shape = tf.concat(0, [original_shape, (output_dim,)])
        xw_plus_b = tf.nn.bias_add(tf.matmul(tf.reshape(xt,[-1,input_dim]), 
                                             Wt), bt)
        xw_plus_b = tf.reshape(xw_plus_b, new_shape)
        if activation == None or activation == 'linear':
            out = xw_plus_b
        else:
            out = activation(xw_plus_b)
        return out, [W], [b]

def build_rnn_inputs(dim_input, dim_output):
    nn_input = tf.placeholder(tf.float32, shape=(None, None, dim_input),
                              name='nn_input')
    action = tf.placeholder(tf.float32, shape=(None, None, dim_output),
                            name='action')
    precision = tf.placeholder(tf.float32,
                               shape=(None, None, dim_output, dim_output),
                               name='precision')
    return nn_input, action, precision

def build_rnn(input_tensor, dim_input, dim_output, scope='LSTM'):
    initial_state = tf.placeholder(tf.float32, shape=(None, 2*dim_output))


    lstm_cell = tf.nn.rnn_cell.LSTMCell(dim_output, input_size=dim_input, )
    y, states = tf.nn.dynamic_rnn(lstm_cell, input_tensor, 
                                  initial_state=initial_state, 
                                  dtype='float32',
                                  
                                  scope=scope)
    weights, biases = tf.all_variables()
    return y, states, initial_state, [weights], [biases]

def build_crl_ff_layers(rnn_output, nn_input, rnn_output_dim, nn_input_dim,
                        nn_output_dim, hidden_dim=128, n_layers=3):
    weights = []
    biases = []
    first_out, first_W, first_b = time_distributed_multi_input_dense_layer(
                    rnn_output, nn_input, rnn_output_dim, nn_input_dim, 
                    hidden_dim)
    weights += first_W
    biases += first_b

    prev_out = first_out
    for _ in range(n_layers - 2):
        prev_out, prev_W, prev_b  = time_distributed_dense_layer(
                prev_out, hidden_dim, hidden_dim)
        weights += prev_W
        biases += prev_b

    final_out, final_W, final_b = time_distributed_dense_layer(
            prev_out, hidden_dim, nn_output_dim, activation=None)
    weights += final_W
    weights += final_b

    return final_out, weights, biases

def build_loss(rnn_out, action, precision, output_dim):
    return time_distributed_euclidean_loss_layer(action, rnn_out, precision)

def example_rnn_network(dim_input=32, dim_output=7, batch_size=10):
    # The simplest possible rnn: a single cell
    n_steps = 100

    nn_input, action, precision = build_rnn_inputs(dim_input, dim_output)
    rnn_out, rnn_states, rnn_initial_state = build_rnn(nn_input, dim_input,
                                                       dim_output)
    loss_out = build_loss(rnn_out, action, precision)
    
    return TfMap.init_from_lists([nn_input, action, precision],
                                 [rnn_out, rnn_states, rnn_initial_state],
                                 [loss_out], recurrent=True)



def crl_rnn_large(dim_input=32, dim_output=7, batch_size=10,
                    n_steps=100, rnn_output_dim=256, hidden_dim=256,
                    n_feedforward=4, network_config=None):
    ''' Implements the RNN architecture we plan to use for CRL
        :first layer: RNN. Estimates system parameters.
        :remaining layers: Feed forward. Map state + params -> action '''
    print "Constructing CRL RNN Network"
    print("... dim_input = %d, dim_output=%d, rnn_output_dim=%d, hidden_dim=%d"
          %(dim_input, dim_output, rnn_output_dim, hidden_dim))

    nn_input, action, precision = build_rnn_inputs(dim_input, dim_output)
    rnn_out, rnn_states, rnn_initial_state, \
       rnn_weights, rnn_biases = build_rnn(nn_input, dim_input, rnn_output_dim)
    nn_out, nn_weights, nn_biases = build_crl_ff_layers(rnn_out, nn_input, 
                                                  rnn_output_dim, dim_input,
                                                  dim_output, 
                                                  hidden_dim=hidden_dim, 
                                                  n_layers=n_feedforward)
    weights = rnn_weights + nn_weights
    biases = rnn_biases + nn_biases

    loss_out = build_loss(nn_out, action, precision, dim_output)
    return TfMap.init_from_lists([nn_input, action, precision],
                                 [nn_out, rnn_states, rnn_initial_state],
                                 [loss_out], [weights, biases], 
                                 recurrent=True)
def crl_rnn_network(dim_input=32, dim_output=7, batch_size=10,
                    n_steps=100, rnn_output_dim=64, hidden_dim=32,
                    n_feedforward=3, network_config=None):
    ''' Implements the RNN architecture we plan to use for CRL
        :first layer: RNN. Estimates system parameters.
        :remaining layers: Feed forward. Map state + params -> action '''
    print "Constructing CRL RNN Network"
    print("... dim_input = %d, dim_output=%d, rnn_output_dim=%d, hidden_dim=%d"
          %(dim_input, dim_output, rnn_output_dim, hidden_dim))

    nn_input, action, precision = build_rnn_inputs(dim_input, dim_output)
    rnn_out, rnn_states, rnn_initial_state, \
       rnn_weights, rnn_biases = build_rnn(nn_input, dim_input, rnn_output_dim)
    nn_out, nn_weights, nn_biases = build_crl_ff_layers(rnn_out, nn_input, 
                                                  rnn_output_dim, dim_input,
                                                  dim_output, 
                                                  hidden_dim=hidden_dim, 
                                                  n_layers=n_feedforward)
    weights = rnn_weights + nn_weights
    biases = rnn_biases + nn_biases

    loss_out = build_loss(nn_out, action, precision, dim_output)
    return TfMap.init_from_lists([nn_input, action, precision],
                                 [nn_out, rnn_states, rnn_initial_state],
                                 [loss_out], [weights, biases], 
                                 recurrent=True)
