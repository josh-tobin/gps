from __future__ import division

import datetime
import numpy as np

from agent.mjc.agent_mjc import AgentMuJoCo
from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from algorithm.cost.cost_state import CostState
from algorithm.cost.cost_torque import CostTorque
from algorithm.cost.cost_sum import CostSum

from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from algorithm.policy.lin_gauss_init import init_lqr, init_pd
from sample_data.gps_sample_types import *

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
    'dX': 26,
    'dU': 7,
    'dO': 26,
    'state_include': [JointAngles, JointVelocities, EndEffectorPoints, EndEffectorPointVelocities],
    #'state_include': [JointAngles, JointVelocities],
    'obs_include': [],
    # TODO - Have sample data compute this, and instead feed in the dimensionalities of each sensor
    'state_idx': [list(range(7)), list(range(7, 14)), list(range(14, 20)), list(range(20, 26))],
    #'state_idx': [list(range(7)), list(range(7, 14))],
    'obs_idx': [],
}

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/pr2_arm3d.xml',
    'frozen_joints': [7, 8, 9, 10],  # Freeze fingertips
    #'init_pose': [0,0,0,0,0,0,0,0.5,0.5,0.5,0.5],
    'init_pose': [0.1,0.1,-1.54,-1.7,1.54,-0.2,0,0.5,0.5,0.5,0.5],
    'rk': 1,
    'dt': 0.01,
    'substeps': 1
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': 1,
}

PR2_GAINS = np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])
algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'args': {
        'hyperparams': {
            'init_gains':  (1.0)/PR2_GAINS,
            'init_acc': np.zeros(sample_data['dU']),
            'init_var': 5.0,
            'init_stiffness': 1.0,
            'init_stiffness_vel': 0.5
            },
        'dt': agent['dt'],
    }
}

torque_cost = {
    'type': CostTorque,
    'wu': 5e-3*PR2_GAINS
}
state_cost = {
    'type': CostState,
    'data_types' : {
        JointAngles: {
            'wp': np.array([1,1,1,1,1,1,1]),
            'desired_state': np.array([0.0,-0.0,0.,0,0,0.,1.0])
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost],
    'weights': [0.0, 1.0]
}

algorithm['dynamics'] = {
    'type': DynamicsLR,
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

defaults = {
    'iterations': 10,
    'common': common,
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}
