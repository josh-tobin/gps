import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np
import imp
from gps.agent.ros.agent_ros import AgentROS
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment')
    args = parser.parse_args()
    hyperparams_file = '/home/jt/gps/experiments/' + args.experiment + '/hyperparams.py'
    test_openloop(hyperparams_file)

def test_openloop(hyperparams_file):
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    agent = hyperparams.config['agent']['type'](hyperparams.config['agent'])
    T = agent.T
    dU = agent.dU
    dX = agent.dX
    K = np.zeros([T, dU, dX])
    k = np.ones([T, dU])
    pol_covar = np.zeros([T, dU, dU])
    chol_pol_covar = np.zeros([T, dU, dU])
    inv_pol_covar = np.zeros([T, dU, dU])
    policy = LinearGaussianPolicy(K, k, pol_covar, chol_pol_covar, 
                                  inv_pol_covar)
    cost = hyperparams.config['algorithm']['cost']['type'](hyperparams.config['algorithm']['cost'])
    sample = agent.sample(policy, 0, verbose=True, save=False)
    print "--------------------------------------------------"
    print sample.get_X()
if __name__ == '__main__':
    main()
