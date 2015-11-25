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

from gps.gui.target_setup import load_from_npz

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

common = {
    'conditions': 1,
    'experiment_dir': BASE_DIR + '/experiments/default_pr2_experiment/',
    'experiment_name': 'my_experiment_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
}
common['target_files_dir'] = common['experiment_dir'] + 'target_files/'
common['output_files_dir'] = common['experiment_dir'] + 'output_files/'

if not os.path.exists(common['target_files_dir']):
    os.makedirs(common['target_files_dir'])
if not os.path.exists(common['output_files_dir']):
    os.makedirs(common['output_files_dir'])

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

ja_x0  = load_from_npz(common['target_files_dir'] + 'trial_arm_initial.npz', 'ja0', default_dim=7)
ee_pos_x0  = load_from_npz(common['target_files_dir'] + 'trial_arm_initial.npz', 'ee_pos0', default_dim=3)
ee_rot_x0  = load_from_npz(common['target_files_dir'] + 'trial_arm_initial.npz', 'ee_rot0', default_dim=9)[0]

ja_tgt = load_from_npz(common['target_files_dir'] + 'trial_arm_target.npz', 'ja0', default_dim=7)
ee_pos_tgt  = load_from_npz(common['target_files_dir'] + 'trial_arm_target.npz', 'ee_pos0', default_dim=3)
ee_rot_tgt  = load_from_npz(common['target_files_dir'] + 'trial_arm_target.npz', 'ee_rot0', default_dim=9)[0]

# TODO - construct this somewhere else?
x0 = np.zeros(23)
x0[0:7] = ja_x0
x0[14:] = np.ndarray.flatten(get_ee_points(ee_points, ee_pos_x0, ee_rot_x0))

ee_tgt = np.ndarray.flatten(get_ee_points(ee_points, ee_pos_tgt, ee_rot_tgt))

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
     #TODO: Controller will seg fault when passed in empty points. For now just use at least one point (0,0,0)
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

from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
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

defaults = {
    'iterations': 20,
    'common': common,
    'agent': agent,
    # 'gui': gui,
    'algorithm': algorithm,
    'num_samples': 5,
}
