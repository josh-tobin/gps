""" Hyperparameters for PR2 policy optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_ros import AgentROS
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.target_setup_gui import load_pose_from_npz
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
        TRIAL_ARM, AUXILIARY_ARM, JOINT_SPACE
from gps.utility.general_utils import get_ee_points
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import example_tf_network


BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])

####################
# THINGS TO CHANGE #
####################

N_DOFS = 1
DOFS = [3]

CONDITIONS = 1
STARTING_POS = [np.array([-0.58687217])]
TARGET_POS = np.array([-0.25])

EXP_DIR = BASE_DIR + '/../experiments/xfer_1arm_ff/'

################################
# PROBABLY DONT NEED TO CHANGE #
################################
EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, 0.05],
                      [0.02, 0.05, 0.0]])

SENSOR_DIMS = {
    JOINT_ANGLES: N_DOFS,
    JOINT_VELOCITIES: N_DOFS,
    ACTION: N_DOFS,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
}


x0 = np.array([0.40493573, -0.26709164, 0.9860822, -0.58687217, -30.2900127,
        -0.59461712, 19.50285032])
x0 = np.concatenate([x0, np.zeros(7)])
x0s = [x0] * common['conditions']
for i, x in enumerate(x0s):
    for dof, pos in zip(DOFS, STARTING_POS[i]):
        x[dof] = pos

tgt = x0.copy()
for dof, pos in zip(DOFS, TARGET_POS):
    tgt[dof] = pos
tgts = [tgt] * common['conditions']

aux_x0 = np.zeros(7)

reset_condition = {
    TRIAL_ARM: {
        'mode': JOINT_SPACE,
        'data': x0[:7],
    },
    AUXILIARY_ARM: {
        'mode': JOINT_SPACE,
        'data': aux_x0,
    },
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentROS,
    'dt': 0.05,
    'conditions': common['conditions'],
    'T': 100,
    'x0': x0s,
    'ee_points_tgt': np.zeros(9),
    'reset_conditions': reset_condition,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 5,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-1]),
    'policy_dual_rate': 0.1,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 5.0,
    'init_pol_wt': 0.01,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
    'max_policy_samples': 6,
    'policy_sample_mode': 'add',
    'save_samples': True,
    'x0': [np.concatenate([x[DOFS], x[[dof + 7 for dof in DOFS]]]) 
                            for x in x0s],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS[DOFS],
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 0.5,
    'stiffness_vel': 0.25,
    'final_weight': 50,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-3 / PR2_GAINS[DOFS],
}

state_cost = {
    'type': CostState,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': tgts,
            'wp': np.ones(N_DOFS),
        }
    },
    'l1': 0.1,
    'l2': 1.0,
    'alpha': 1e-5,
}

final_cost = {
    'type': CostState,
    'ramp_option': RAMP_FINAL_ONLY,
    'data_types': {
        JOINT_ANGLES: {
            'target_state': tgts,
            'wp': state_cost['data_types'][JOINT_ANGLES]['wp'],
        }
    },
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1e-5,
    'wp_final_multiplier': 10.0,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, state_cost, final_cost],
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

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
    },
    'low_dof': True,
    'dofs': DOFS,
    'network_model': example_tf_network,
    'iterations': 1000,
    'weights_file_prefix': EXP_DIR + 'policy',
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': algorithm['iterations'],
    'common': common,
    'verbose_trials': 0,
    'verbose_policy_trials': 1,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

common['info'] = generate_experiment_info(config)
