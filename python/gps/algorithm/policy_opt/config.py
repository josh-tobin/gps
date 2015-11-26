import caffe
import numpy as np

from policy_opt_caffe_util import construct_fc_network

policy_opt_caffe = {
    # Initialization
    'init_var': 0.1,  # Initial policy variance
    'ent_reg': 0.0,  # Entropy regularizer
    # Solver hyperparameters
    'iterations': 20000,  # Number of iterations of training per inner iteration
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed)
    'lr_policy': 'fixed',  # Learning rate policy
    'momentum': 0.9,  # momentum
    'weight_decay': 0.005,  # weight decay
    'use_gpu': 1,  # Whether or not to use the gpu for caffe training
    'gpu_id': 0,
    'solver_type': 'Adam',  # Solver type to use (e.g. 'SGD', 'Adam', etc.)
    # Other hyperparameters
    'network_model': construct_fc_network,  # Either a filename string or a function to call to create NetParameter
    'network_arch_params': {},  # Arguments to pass to network construction method above
    'weights_file_prefix': '',
    'rate_schedule': np.array([1, 1]),
    'rate_schedule_end': 9,
}
