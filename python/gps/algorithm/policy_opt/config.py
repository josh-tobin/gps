import caffe
from policy_opt_caffe_util import construct_fc_network

policy_opt_caffe = {
    # Solver hyperparameters
    'iterations': 500,  # Number of iterations of training per inner iteration
    'batch_size': 25,
    'lr': 0.001,  # Base learning rate (by default it's fixed) for SGD
    'use_gpu': 1,  # Whether or not to use the gpu for caffe training
    'gpu_id': 0,
    'solver_type': 'SGD',  # Solver type to use (e.g. 'SGD', 'Adam', etc.)
    # Other hyperparameters
    'network_model': construct_fc_network,  # Either a filename string or a function to call to create NetParameter
    'network_arch_params': {},  # Arguments to pass to network construction method above
    'weights_file_prefix': '',
}
