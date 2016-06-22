""" Hyperparameters for MJC peg insertion policy optimization. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
#from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import example_tf_network
from gps.utility.general_utils import get_ee_points
from gps.gui.target_setup_gui import load_pose_from_npz

from gps.utility.general_utils import find_bodies_in_model, find_objects_in_model

from gps.algorithm.policy_opt.tf_rnn import crl_rnn_network, crl_rnn_large


BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])

#############
# TO CHANGE #
#############
MASS_RANGE = np.array([0.5, 1.5])
GAIN_RANGE = np.array([0.5, 1.5])
#GAIN_RANGE = np.array([0.5, 1.5])
OBJ_MASS_RANGE = np.array([0.05, 0.5])

EXP_DIR = BASE_DIR + '/../experiments/crl_jar_svv40/'

MODEL = crl_rnn_large

CONDITIONS = 40
SAMPLES_PER_CONDITION = 5

MODEL_PATH = './mjc_models/PR2/pr2_jar.xml' 

###############
# DONT CHANGE #
###############

EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, 0.05],
                  [0.02, 0.05, 0.0]])

body_names = ['l_shoulder_pan_link', 'l_shoulder_lift_link', 
              'l_upper_arm_roll_link', 'l_upper_arm_link',
              'l_elbow_flex_link', 'l_forearm_roll_link',
              'l_forearm_link']

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    ACTION: 7,
}

PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])


x0s = []
ee_tgts = []
reset_conditions = []

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
}

#for i in xrange(common['conditions']):
for i in range(1):
    ja_x0, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
            common['target_filename'], 'trial_arm', str(i), 'initial'
    )
    ja_x0_tgt, ee_pos_tgt, ee_rot_tgt = load_pose_from_npz(
            common['target_filename'], 'trial_arm', str(i), 'target'
    )

    x0 = np.zeros(32)
    x0[:7] = ja_x0
    eepts =  np.ndarray.flatten(
            get_ee_points(EE_POINTS, ee_pos_x0, ee_rot_x0).T)
    x0[14:(14+3*EE_POINTS.shape[0])] = eepts

    ee_tgt = np.ndarray.flatten(
            get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
    )

    x0s.append(x0)
    # For reaching experiment
    ee_tgts.append(ee_tgt)
    # For holdarm experiment, tgt is the same as x0
    #ee_tgts.append(eepts)

x0s*= common['conditions']
ee_tgts *= common['conditions']

#print 'ee tgt is'
#print ee_tgt

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

bodies = find_bodies_in_model(MODEL_PATH, body_names)
objects = find_objects_in_model(MODEL_PATH)
#all_bodies = bodies.update(objects)
body_idx = bodies.values()
object_idx = objects.values()
num_bodies = len(body_idx)
num_objects = len(object_idx)
agent = {
    'type': AgentMuJoCo,
    'filename': MODEL_PATH,
    'x0': [x[:14] for x in x0s],
    'dt': 0.05,
    'substeps': 50,
    'ee_points_tgt': ee_tgts,
    'conditions': common['conditions'],
    'mass_body_idx': np.concatenate([body_idx, object_idx]),
    'mass_body_mult': [np.concatenate([
        np.random.uniform(low=MASS_RANGE[0], 
                          high=MASS_RANGE[1], size=num_bodies),
        np.random.uniform(low=OBJ_MASS_RANGE[0],
                          high=OBJ_MASS_RANGE[1], size=num_objects)])
                       for _ in range(common['conditions'])],
    'gain_scale': [
        np.random.rand(7) * (GAIN_RANGE[1] - GAIN_RANGE[0]) 
        for _ in range(common['conditions'])
    ],
    'pos_body_idx': np.array([1]),
    'pos_body_offset': np.array([0.0, 0.0, 0.0]),
    'body_color_offset': np.array([1]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES, ACTION],
    'camera_pos': np.array([3., 0.5, 4., 0.25, 0, 0.5]),
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 10,
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    'kl_step': 2.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'policy_sample_mode': 'replace',
    'save_samples': False,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / PR2_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0, # vs 1.0 for std experiment
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-3 / PR2_GAINS,
}

#pose_target = np.array([0.0, 0.3, -0.35])
#ee_pts_tgt = (EE_POINTS + pose_target).flatten()
#ee_pts_tgt = np.array([0.886, -0.0484, -0.201, 0.886, -0.0484, -0.201, 
#                       0.886, -0.0508, -0.272])
#print "ja_tgt is: " + str(ja_x0_tgt)
ee_pts_tgt = np.zeros([9,])
fk_cost = {
    'type': CostFK,
    'target_end_effector': ee_pts_tgt,
    'wp': np.ones(3*EE_POINTS.shape[0]),
    'l1': 0.1,
    'l2': 1.0, #10.0
    'alpha': 1e-5,
}

# Create second cost function for last step only.
final_cost = {
    'type': CostFK,
    'ramp_option': RAMP_FINAL_ONLY,
    'target_end_effector': fk_cost['target_end_effector'],
    'wp': fk_cost['wp'],
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1e-5,
    'wp_final_multiplier': 10.0,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost, final_cost],
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

#algorithm['policy_opt'] = {
#    'type': PolicyOptCaffe,
#    'weights_file_prefix': EXP_DIR + 'policy',
#    'iterations': 2000,
#}
algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, ACTION],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, ACTION],
        'sensor_dims': SENSOR_DIMS,
    },
    'recurrent': True,
    'network_model': MODEL,
    'iterations': 350,
    'weights_file_prefix': EXP_DIR + 'policy',
    'checkpoint_prefix': EXP_DIR + 'data_files/policy_checkpoint.ckpt',
    'batch_size': 10,
    'use_gpu': True,
}

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': SAMPLES_PER_CONDITION,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
