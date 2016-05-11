import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np
import imp
from gps.agent.ros.agent_ros import AgentROS
from gps.utility.data_logger import DataLogger

def main():
    hyperparams_file = 'experiments/pr2_tensorflow_example/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    agent = AgentROS(hyperparams.config['agent'])
    algo_file = '/home/jt/gps/experiments/pr2_mjc_test/data_files/algorithm_itr_09.pkl'
    data_logger = DataLogger()
    algorithm = data_logger.unpickle(algo_file)
            

#def test_policy(agent, policy, conditions, algo_file, N):



if __name__ == '__main__':
    main()
