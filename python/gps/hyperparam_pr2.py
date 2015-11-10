from __future__ import division

from datetime import datetime
import numpy as np
import os.path

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_ros import AgentROS
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_torque import CostTorque
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr import DynamicsLR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.proto.gps_pb2 import *


SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}

PR2_GAINS = np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-3])

common = {
    'conditions': 1,
    'experiment_dir': BASE_DIR + '/experiments/default_pr2_experiment/',
    'experiment_name': 'my_experiment_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}
common['target_files_dir'] = common['experiment_dir'] + 'target_files/'
common['output_files_dir'] = common['experiment_dir'] + 'output_files/'

if not os.path.exists(common['experiment_dir']):
    os.makedirs(common['experiment_dir'])
if not os.path.exists(common['target_files_dir']):
    os.makedirs(common['target_files_dir'])
if not os.path.exists(common['output_files_dir']):
    os.makedirs(common['output_files_dir'])

x0 = np.zeros(14)  # Assume initial state should have 0 velocity
filename = common['target_files_dir'] + 'trial_arm' + '_initial.npz'
try:
    with np.load(filename) as f:
        x0[0:7] = f['ja0']
except IOError as e:
    print('No initial file found, defaulting to all zeros state')

tgt = np.zeros(7)  # Assume target state should have 0 velocity
filename = common['target_files_dir'] + 'trial_arm' + '_target.npz'
try:
    with np.load(filename) as f:
        tgt = f['ja0']
except IOError as e:
    print('No target file found, defaulting to all zeros state')


agent = {
    'type': AgentROS,
    'dt': 0.05,
    'x0': x0,
    'conditions': common['conditions'],
    'T': 100,
    'reset_conditions': {
        0: {
            TRIAL_ARM: {
                'mode': JOINT_SPACE,
                'data': x0[0:7],
                #'data': np.array([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5])
            },
            AUXILIARY_ARM: {
                'mode': JOINT_SPACE,
                'data': np.zeros(7),
            },
        },
     },
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'args': {
        'hyperparams': {
            'init_gains':  1.0 / PR2_GAINS,
            'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
            'init_var': 1.0,
            'init_stiffness': 1.0,
            'init_stiffness_vel': 0.5,
        },
        'dt': agent['dt'],
        'T': agent['T'],
        'x0': agent['x0'],
    }
}

torque_cost = {
    'type': CostTorque,
    'wu': 5e-3/PR2_GAINS,
}

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            #'desired_state': np.array([0.617830101225870, 0.298009357128493, -2.26613599619067,
                #-1.83180464491005, 1.44102734751961, -0.488554457910043, -0.311987910094871]),
            #'desired_state': np.array([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5]),
            'desired_state': tgt,
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost],
    'weights': [1.0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

defaults = {
    'iterations': 20,
    'common': common,
    'agent': agent,
    # 'gui': gui,
    'algorithm': algorithm,
    'num_samples': 5,
}
