from __future__ import division

from math import pi
from datetime import datetime
import numpy as np
import os.path

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.arm_world import ArmWorld
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_torque import CostTorque
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr import DynamicsLR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.proto.gps_pb2 import *

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    ACTION: 2
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-3])
EXP_DIR = BASE_DIR + '/experiments/bod2d_arm_experiment/'


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([1.75*pi, 1.5*pi]),
    "world" : ArmWorld,
    'x0': np.array([0, 0, 0, 0]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 1, #5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    # TODO - incorporate pos_body_offset into box2d agent
    'pos_body_offset': np.array([]), #[np.array([0, 0.2, 0]), np.array([0, 0.1, 0]), np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    'T': 100,
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
    'init_gains':  np.array([0, 0]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'init_stiffness': 0,
    'init_stiffness_vel': 0.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostTorque,
    'wu': np.array([5e-5,5e-5])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.array([1, 1]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost],
    'weights': [0, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': 10,
    'num_samples': 5,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui': False,
    'algorithm': algorithm,
}
