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

#EXP_FILE = '/home/jt/gps/experiments/'
EXP_FILE = os.path.dirname(os.path.realpath(__file__))
EXP_FILE += 'experiments/'
HYPERPARAMS_FILE = '/hyperparams.py'
POLICY_FILE = '/data_files/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparams_file')
    parser.add_argument('policy_file')
    parser.add_argument('policy_iter', type=int)
    parser.add_argument('--v', type=bool, default=False)
    args = parser.parse_args()

    collect_sample(args.hyperparams_file, args.policy_file, args.policy_iter)


def collect_sample(exp_name, policy_name, policy_iter):
    tester = PolicyTester(exp_name)
    tester.test(policy_name, policy_iter, verbose=True)
'''
def collect_sample(exp_name, policy_name, policy_iter):
    hyperparams_file = EXP_FILE + exp_name + HYPERPARAMS_FILE
    #hyperparams_file = 'experiments/pr2_gazebo_xfer/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    agent = hyperparams.config['agent']['type'](hyperparams.config['agent'])
    
    #algo_file = '/home/jt/gps/experiments/pr2_mjc_test/data_files/algorithm_itr_09.pkl'
    algo_file = EXP_FILE + policy_name + POLICY_FILE + \
            'algorithm_itr_%02d.pkl'%policy_iter

    data_logger = DataLogger()
    algorithm = data_logger.unpickle(algo_file)
    pol = algorithm.policy_opt.policy
    cost = hyperparams.config['algorithm']['cost']['type'](hyperparams.config['algorithm']['cost'])
    
    
    C = hyperparams.config['common']['conditions']
    
    if args.v:
        print "------------------------------------------------"
    costs = []
    for c in range(C):
        sample = agent.sample(pol, c, verbose=True, save=False)
        l, _, _, _, _, _ = cost.eval(sample)
        if args.v:
            print "Condition %d: cost sum is %f"%(c, np.sum(l))
            print
'''

if __name__ == '__main__':
    main()
