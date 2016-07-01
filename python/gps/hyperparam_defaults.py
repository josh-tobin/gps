from __future__ import division

import datetime
import os
import numpy as np

from agent.mjc.agent_mjc import AgentMuJoCo
from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from algorithm.cost.cost_fk import CostFK
from algorithm.cost.cost_state import CostState
from algorithm.cost.cost_torque import CostTorque
from algorithm.cost.cost_sum import CostSum

from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from proto.gps_pb2 import *

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
GPS_ROOT_DIR = os.path.dirname(os.path.dirname(THIS_FILE_DIR))

common = {
    'conditions': 4,
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
    'dX': 26,
    'dU': 7,
    'dO': 26,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
    # TODO - Have sample data compute this, and instead feed in the dimensionalities of each sensor
    'state_idx': [list(range(7)), list(range(7, 14)), list(range(14, 20)), list(range(20, 26))],
    'obs_idx': [],
}

"""
agent = {
    'type': AgentMuJoCo,
    'filename': os.path.join(GPS_ROOT_DIR, 'mjc_models/pr2_arm3d.xml'),
    'init_pose': np.concatenate([np.array([0.1,0.1,-1.54,-1.7,1.54,-0.2,0]), np.zeros(7)]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [np.array([-0.5,0.2,0.1]), np.array([0,0.0,-0.2]), np.array([-0.4,0.1,0.1]), np.array([0,0.0,0]), np.array([0.1,-0.2,0]), np.array([-0.1,0.0,0.2]),
    np.array([-0.3,0.1,0.1]), np.array([-0.2,0.3,0]), np.array([-0.3,0.8,0.2]), np.array([0.5,0.5,0.1])]
}
"""
SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}
agent = {
    'type': AgentMuJoCo,
    'filename': os.path.join(GPS_ROOT_DIR, 'mjc_models/pr2_arm3d.xml'),
    'x0': np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]),
                          np.zeros(7)]),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': [np.array([0, 0.2, 0]), np.array([0, 0.1, 0]),
                        np.array([0, -0.1, 0]), np.array([0, -0.2, 0])],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
}


algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

PR2_GAINS = np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])
algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'args': {
        'hyperparams': {
            'init_gains':  1.0/PR2_GAINS,
            'init_acc': np.zeros(sample_data['dU']),
            'init_var': 1.0,
            'init_stiffness': 1.0,
            'init_stiffness_vel': 0.5
            },
        'dt': agent['dt'],
    }
}

torque_cost = {
    'type': CostTorque,
    'wu': 5e-5/PR2_GAINS
}
state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(sample_data['dU']),
            # This should extend the arm out straight
            #'desired_state': np.array([0.0,0.,0.,0.,0.,0.,0.])

            # This should insert into the hold
            'desired_state': np.array([0.617830101225870,0.298009357128493,-2.26613599619067,-1.83180464491005,1.44102734751961,-0.488554457910043,-0.311987910094871])
        },
    },
}

fk_cost = {
    'type': CostFK,
    'end_effector_target': np.array([0.0, 0.3, -0.5,  0.0, 0.3, -0.2]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
    'wp': np.array([1,1,1,1,1,1]),
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost],
    'weights': [1.0, 1.0]
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

defaults = {
    'iterations': 20,
    'common': common,
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}