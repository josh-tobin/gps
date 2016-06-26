import sys
sys.path.append('python')
import numpy as np
import imp
from gps.agent.ros.agent_ros import AgentROS
from gps.utility.data_logger import DataLogger
from gps.agent.ros.ros_utils import msg_to_sample, policy_to_msg
from gps_agent_pkg.msg import TrialCommand
from gps.agent.agent_utils import generate_noise
from saved_examples.util import PolicyTester
import argparse
import os
import cPickle as pickle

#EXP_FILE = '/home/jt/gps/experiments/'
EXP_FILE = os.path.dirname(os.path.realpath(__file__))
EXP_FILE += '/experiments/'
HYPERPARAMS_FILE = '/hyperparams.py'
POLICY_FILE = '/data_files/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparams_file')
    parser.add_argument('policy_file')
    parser.add_argument('--policy_iter', type=int, default=9)
    parser.add_argument('--v', type=bool, default=False)
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()

    collect_sample(args.hyperparams_file, args.policy_file, args.policy_iter,
                   n_samples=args.n_samples)

def collect_sample(exp_name, policy_name, policy_iter, n_samples=5):
    tester = PolicyTester(exp_name)
    err = tester.test(policy_name, policy_iter, verbose=True, n_samples=n_samples)
    error_data_file = EXP_FILE + policy_name + POLICY_FILE + 'error_data.pkl'
    try:
        with open(error_data_file, 'r') as f:
            err_data = pickle.load(f)
    except:
        err_data = {'train_error': {}, 'test_error': {}}
    err_data['test_error'][exp_name] = err
    with open(error_data_file, 'wb') as f:
        pickle.dump(err_data, f)

if __name__ == '__main__':
    main()
