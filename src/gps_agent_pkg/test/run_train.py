import time
import os
import numpy as np
import scipy
import scipy.io
import logging
import argparse
import cPickle
from hyperparam_defaults import defaults
from sample_data.sample_data import SampleData
from algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from agent.ros.agent_ros import AgentROS
from proto.gps_pb2 import *
import rospy

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.debug("Test debug message")
np.set_printoptions(suppress=True)
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_agent(T=20):
    defaults['sample_data']['T'] = T
    defaults['sample_data']['state_include'] = [JOINT_ANGLES, JOINT_VELOCITIES]
    sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
    agent = AgentROS(defaults['agent'], sample_data)
    r = rospy.Rate(1)
    r.sleep()
    return sample_data, agent

def run_offline():
    """
    Run offline controller, and save results to controllerfile
    """
    sample_data, agent = setup_agent()
    algorithm = defaults['algorithm']['type'](defaults['algorithm'], sample_data)
    conditions = 4
    idxs = [[] for _ in range(conditions)]
    for itr in range(10): # Iterations
        for m in range(conditions):
            for i in range(2): # Trials per iteration
                n = sample_data.num_samples()
                pol = algorithm.cur[m].traj_distr
                sample = agent.sample(pol, sample_data.T, m)
                #TODO: Reset
                sample_data.add_samples(sample)
                idxs[m].append(n)
        algorithm.iteration([idx[-20:] for idx in idxs])
        print 'Finished itr ', itr
    import pdb; pdb.set_trace()

run_offline()
