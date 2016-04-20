from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed

import tensorflow as tf
from gps.algorithm.policy_opt.tf_model_example import get_loss_layer, \
                                               batched_matrix_vector_multiply
from gps.algorithm.policy_opt.tf_utils import TfMap

class KerasPolicyContainer(object):
    ''' A class to contain a keras policy. Can be passed to TfMap to 
        get a TfMap object that is then used to create a TfPolicy '''
    def __init__(self, dim_input=27, dim_output=7, batch_size=25,
                 n_layers=3):
        raise NotImplementedError

def get_inputs(dim_input, dim_output):
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], 
                                name='precision')
    return action, precision

def time_distributed_euclidean_loss_layer(a, b, precision, 
                                          batch_size, max_timesteps):
    scale_factor = tf.constant(2*batch_size*max_timesteps, dtype='float')
    u_l_exp = tf.expand_dims(a-b, -2)
    uP = tf.batch_matmul(u_l_exp, precision)
    u_r_exp = tf.expand_dims(a-b, -1)
    uPu = tf.reduce_sum(tf.batch_matmul(uP, u_r_exp))
    print(uPu.dtype)
    print(scale_factor.dtype)
    return uPu/scale_factor

def get_rnn_loss_layer(mlp_out, action, precision, batch_size, max_timesteps):
    return time_distributed_euclidean_loss_layer(a=action, b=mlp_out,
                                                 precision=precision,
                                                 batch_size=batch_size,
                                                 max_timesteps=max_timesteps)

def get_rnn_inputs(dim_input, dim_output):
    action = tf.placeholder('float', [None, None, dim_output], name='action')
    precision = tf.placeholder('float', [None, None, dim_output, dim_output],
                               name='precision')
    return action, precision

def example_keras_network(dim_input=27, dim_output=27, batch_size=25):
    n_layers = 3
    dim_hidden = (n_layers - 1) * [42]
    dim_hidden.append(dim_output)

    model = Sequential()
    model.add(Dense(input_dim=dim_input, output_dim=dim_hidden[0], 
                    activation='relu'))
    for i in range(n_layers-2):
        model.add(Dense(input_dim=dim_hidden[i], output_dim=dim_hidden[i+1],
                        activation='relu'))
    model.add(Dense(input_dim=dim_hidden[-1], output_dim=dim_output,
                    activation='linear'))

    nn_input = model.get_input()
    action, precision = get_inputs(dim_input, dim_output)
    mlp_applied = model.get_output()
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, 
            precision=precision, batch_size=batch_size)
    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied],
                                 [loss_out])

def keras_rnn_network(dim_input=27, dim_output=7, batch_size=10):
    n_recurrent = 1
    recurrent_size = 64
    n_hidden = 2
    hidden_size = 64
    max_path_length = 20
    
    model = Sequential()
    
    model.add(LSTM(recurrent_size, 
                   batch_input_shape=(batch_size, max_path_length, dim_input),
                   return_sequences=True, stateful=True))
    for _ in range(n_recurrent - 1):
        model.add(LSTM(recurrent_size, return_sequences=True, 
                        stateful=True))

    for _ in range(n_hidden):
        model.add(TimeDistributed(Dense(hidden_size, activation='relu')))

    model.add(TimeDistributed(Dense(dim_output, activation='relu')))

    nn_input = model.get_input()
    action, precision = get_rnn_inputs(dim_input, dim_output)
    mlp_applied = model.get_output()
    loss_out = get_rnn_loss_layer(mlp_applied, action, precision, batch_size,
                                  max_path_length)
    recurrent_reset_states = model.reset_states
    return TfMap.init_from_lists([nn_input, action, precision], [mlp_applied],
                                 [loss_out], recurrent=True, 
                                 recurrent_reset_states=recurrent_reset_states)

