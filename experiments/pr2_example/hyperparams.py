from __future__ import division

from datetime import datetime
import numpy as np
import os.path

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_ros import AgentROS
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_torque import CostTorque
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import *


# TODO - Put this somewhere else.
def get_ee_points(offsets, ee_pos, ee_rot):
    """
    Helper method for computing the end effector points given a position, rotation matrix, and
    offsets for each of the ee points.

    Args:
        offsets: N x 3 array where N is the number of points.
        ee_pos: 1 x 3 array of the end effector position.
        ee_rot: 3 x 3 rotation matrix of the end effector.
    Returns:
        3 x N array of end effector points.
    """
    return ee_rot.dot(offsets.T) + ee_pos.T


EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, 0.05], [0.02, 0.05, 0.0]])

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    ACTION: 7,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-3])
EXP_DIR = BASE_DIR + '/experiments/pr2_example/'

JA_x0, EE_POS_x0, EE_ROT_x0 = load_pose_from_npz(common['target_filename'], 'trial_arm', '0', 'initial')
_, EE_POS_TGT, EE_ROT_TGT = load_pose_from_npz(common['target_filename'], 'trial_arm', '0', 'target')

# TODO - Construct this somewhere else?
x0 = np.zeros(23)
x0[0:7] = JA_x0
x0[14:] = np.ndarray.flatten(get_ee_points(EE_POINTS, EE_POS_x0, EE_ROT_x0).T)

EE_TGT = np.ndarray.flatten(get_ee_points(EE_POINTS, EE_POS_TGT, EE_ROT_TGT).T)

RESET_CONDITION = {
    TRIAL_ARM: {
        'mode': JOINT_SPACE,
        'data': x0[0:7],
    },
    AUXILIARY_ARM: {
        'mode': JOINT_SPACE,
        'data': np.zeros(7),
    },
}


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

agent = {
    'type': AgentROS,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 100,
    'x0': x0,
    'reset_conditions': {
        0: RESET_CONDITION,
     },
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'end_effector_points': EE_POINTS,
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 10,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'init_stiffness': 1.0,
    'init_stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostTorque,
    'wu': 5e-3/PR2_GAINS,
}

fk_cost1 = {
    'type': CostFK,
    'target_end_effector': EE_TGT,
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 0.1,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
}

# TODO - This isn't quite right.
fk_cost2 = {
    'type': CostFK,
    'target_end_effector': EE_TGT,
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
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
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

common['info'] = (
    'exp_name: '   + str(common['experiment_name'])              + '\n'
    'alg_type: '   + str(algorithm['type'].__name__)             + '\n'
    'alg_dyn:  '   + str(algorithm['dynamics']['type'].__name__) + '\n'
    'alg_cost: '   + str(algorithm['cost']['type'].__name__)     + '\n'
    'iterations: ' + str(config['iterations'])                   + '\n'
    'conditions: ' + str(algorithm['conditions'])                + '\n'
    'samples:    ' + str(config['num_samples'])                  + '\n'
)
