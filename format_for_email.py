import cPickle as pickle
import numpy as np
import argparse
import imp

import time

DATA_LOCATION = 'home/jt/gps/experiments/'

def format_data(exp_name):
    hyperparams_loc = DATA_LOCATION + exp_name + '/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    data_loc = DATA_LOCATION + exp_name + exp_name + '/data_files/error_data.pkl'
    with open(data_loc, 'r') as f:
        data = pickle.load(f)

    for exp, err in data['test_error'].iteritems():
        result = exp_name + ', '
        date = time.strftime("%m_%d")
        result += (date + ', ')
        task = ''
        result += (task + ', ')
        if exp[:3] == 'crl':
            domain = 'Mujoco'
        elif exp[:3] == 'xfe':
            domain == 'BRETT'
        result += (domain + ', ')
        nnet_config = ''
        result += (nnet_config + ', ')
        if hyperparams.MASS_RANGE[0] == hyperparams.MASS_RANGE[1]:
            mass_range = ''
        else:
            mass_range = '%.1f - %.1f'%(hyperparams.MASS_RANGE[0], 
                                    hyperparams.MASS_RANGE[1])





