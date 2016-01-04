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
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr import DynamicsLR
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr, init_pd
from gps.proto.gps_pb2 import *

from gps.gui.target_setup_gui import load_position_from_npz

ee_points = np.array([[0.02,-0.025,0.05],[0.02,-0.025,0.05],[0.02,0.05,0.0]])

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3*ee_points.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3*ee_points.shape[0],
    ACTION: 7,
}

PR2_GAINS = np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-3])
EXP_DIR = BASE_DIR + '/experiments/pr2_example/'

common = {
    'experiment_name': 'my_experiment' + '_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# TODO - put this somewhere else
def get_ee_points(offsets, ee_pos, ee_rot):
    """ Helper method for computing the end effector points given a
    position, rotation matrix, and offsets for each of the ee points.

    Args:
        offsets: Nx3 array where N is the number of points
        ee_pos: 1x3 array of the end effector position
        ee_rot: 3x3 rotation matrix of the end effector.
    Returns:
        3xN array of end effector points
    """
    return ee_rot.dot(offsets.T) + ee_pos.T

ja_x0, ee_pos_x0, ee_rot_x0 = load_position_from_npz(common['target_filename'], 'trial_arm', '0', 'initial')
ja_tgt, ee_pos_tgt, ee_rot_tgt = load_position_from_npz(common['target_filename'], 'trial_arm', '0', 'target')

# TODO - construct this somewhere else?
x0 = np.zeros(23)
x0[0:7] = ja_x0
x0[14:] = np.ndarray.flatten(get_ee_points(ee_points, ee_pos_x0, ee_rot_x0).T)

ee_tgt = np.ndarray.flatten(get_ee_points(ee_points, ee_pos_tgt, ee_rot_tgt).T)

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
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'end_effector_points': ee_points,
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
        'dX': sum([SENSOR_DIMS[state_include] for state_include in agent['state_include']]),
        'dU': 7,
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
            'target_state': ja_tgt,
        },
    },
}

fk_cost1 = {
    'type': CostFK,
    'target_end_effector': ee_tgt, #np.array([0.0, 0.0, 0.0,  0.1, 0.2, 0.3]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]), #np.array([1, 1, 1, 1, 1, 1]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

# TODO - this isn't qutie right.
fk_cost2 = {
    'type': CostFK,
    'target_end_effector': ee_tgt, #np.array([0.0, 0.0, 0.0,  0.1, 0.2, 0.3]),
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]), #np.array([1, 1, 1, 1, 1, 1]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep
    'ramp_option': RAMP_FINAL_ONLY,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost1, fk_cost2],
    'weights': [1.0, 1.0, 1.0],
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
    'iterations': 20,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

info = ('experiment_name: '     + str(common['experiment_name'])        + '\n'
        'algorithm_type: '      + str(algorithm['type'])                + '\n'
        'algorithm_dynamics: '  + str(algorithm['dynamics']['type'])    + '\n'
        'algorithm_cost: '      + str(algorithm['cost']['type'])        + '\n'
        'iterations: '          + str(config['experiment_name'])        + '\n'
        'conditions: '          + str(algorithm['conditions'])          + '\n'
        'samples: '             + str(config['num_samples'])            + '\n')
common['info'] = info