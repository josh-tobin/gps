from __future__ import division

from datetime import datetime
import numpy as np
import os.path

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_torque import CostTorque
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr import DynamicsLR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.proto.gps_pb2 import *


SENSOR_DIMS = {
    POSITION: 2,
    ANGLE: 1,
    LINEAR_VELOCITY: 2,
    ANGULAR_VELOCITY: 1,
    ACTION: 1
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-3])

common = {
    'conditions': 4,
    'experiment_dir': BASE_DIR + '/experiments/box2d_experiment/',
    'experiment_name': 'my_experiment_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}
common['output_files_dir'] = common['experiment_dir'] + 'output_files/'

if not os.path.exists(common['output_files_dir']):
    os.makedirs(common['output_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([0, 2]),
    'x0': np.array([0, 2, 3.1415, 0, 0, 0]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
        np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],

    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [POSITION, ANGLE, LINEAR_VELOCITY, ANGULAR_VELOCITY],
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
            'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
            'init_var': 1.0,
            'init_stiffness': 1.0,
            'init_stiffness_vel': 0.5,
        },
        # 'x0': agent['x0'][:SENSOR_DIMS[JOINT_ANGLES]],
        'dt': agent['dt'],
        'T': agent['T'],
    }
}

torque_cost = {
    'type': CostTorque,
    'wu': np.array([0])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        POSITION: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            'target_state': agent["target_state"],
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

config = {
    'iterations': 10,
    'num_samples': 5,
    'common': common,
    'agent': agent,
    # 'gui': gui,  # For sim, we probably don't want the gui right now.
    'algorithm': algorithm,
}
