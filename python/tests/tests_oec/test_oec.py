import os
import os.path
import sys
import numpy as np
#import tensorflow as tf
import imp

# Add gps/python to path
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', ''))
sys.path.append(gps_path)

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
                        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
                        ACTION
from gps.algorithm.dynamics.dynamics_true import DynamicsTrue
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.sample.sample_list import SampleList
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt

test_file = os.path.abspath(os.path.join(gps_path, '../experiments/ilqr_example/hyperparams.py'))
hyperparams = imp.load_source('hyperparams', test_file)

agent = AgentMuJoCo(hyperparams.config['agent'])
dX = 14
dU = agent.dU
T = agent.T
zeros_policy = LinearGaussianPolicy(
                np.zeros([T, dU, dX]),
                np.zeros([T, dU]),
                np.zeros([T, dU, dU]),
                np.zeros([T, dU, dU]),
                np.zeros([T, dU, dU]),
           )


def test_dynamics_true():
    world = agent.worlds[0]
    dynamics = DynamicsTrue(hyperparams.config['algorithm']['dynamics'].update({'world': world}))
    
    sample = agent.sample(zeros_policy, 0, verbose=False) 
    sample_list = SampleList([sample])
    dynamics.fit([sample_list])

def test_ilqr_with_true_dynamics():
    hyperparams.config['algorithm']['agent'] = agent
    algo = AlgorithmTrajOpt(hyperparams.config['algorithm'])
    
    #world = agent.worlds[0]
    #dynamics = DynamicsTrue(hyperparams.config['algorithm']['dynamics'], world)
    dynamics = DynamicsTrue(hyperparams.config['algorithm']['dynamics'].update({'world': world})
    sample = agent.sample(zeros_policy, 0, verbose=False)
    sample_list = SampleList([sample])
    dynamics.fit([sample_list])
    algo.iteration([sample_list])
    x0 = self._hyperparams['x0'][0]
    mj_X = x0
    mj_U = policy.act(mj_X, None, None)
    for _ in range(self._hyperparams['substeps']):

tests = [
        #test_dynamics_true,
        test_ilqr_with_true_dynamics
    ]

if __name__ == '__main__':
    for test in tests:
        test()
    print "Finished: %d tests passed"%len(tests)
