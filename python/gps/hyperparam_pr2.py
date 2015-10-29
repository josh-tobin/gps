from __future__ import division

from datetime import datetime
import numpy as np

#from agent.mjc.agent_mjc import AgentMuJoCo
from agent.ros.agent_ros import AgentROS
from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from algorithm.cost.cost_fk import CostFK
from algorithm.cost.cost_state import CostState
from algorithm.cost.cost_torque import CostTorque
from algorithm.cost.cost_sum import CostSum

from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from algorithm.policy.lin_gauss_init import init_lqr, init_pd
from proto.gps_pb2 import *


SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 6,
    END_EFFECTOR_POINT_VELOCITIES: 6,
    ACTION: 7,
}

PR2_GAINS = np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])


common = {
    'conditions': 1,
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [],
}

agent = {
    'type': AgentROS,
    'dt': 0.05,
    'init_pose': np.zeros(14),#np.concatenate([np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0]), np.zeros(7)]),
    #'init_pose': np.concatenate([np.array([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5]), np.zeros(7)]),
    'conditions': common['conditions'],
    'T': 100,
    'reset_conditions': {
        0: {
            'trial_arm': {
                'mode': 1,
                'data': np.zeros(7),
                #'data': np.array([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5])
            }, 
            'auxiliary_arm': {
                'mode': 0,
                'data': np.zeros(7),
            }, 
        },
     },
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
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
        'x0': np.zeros(14),
        'dX': 14,
        'dU': 7,
        #'x0': np.concatenate([np.array([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5]), np.zeros(7)]),
    }
}

torque_cost = {
    'type': CostTorque,
    'wu': 5e-5/PR2_GAINS,
}

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(SENSOR_DIMS[ACTION]),
            #'desired_state': np.array([0.617830101225870, 0.298009357128493, -2.26613599619067,
                #-1.83180464491005, 1.44102734751961, -0.488554457910043, -0.311987910094871]),
            'desired_state': np.array([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5]),
        },
    },
    'wp': np.array([1, 1, 1, 1, 1, 1]),
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
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}
