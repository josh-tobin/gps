"""Default configuration and hyperparameter values for Agent objects

"""
from copy import deepcopy

from agent.ros.agent_ros import ARM_RIGHT, ARM_LEFT
from gps_agent_pkg.msg import PositionCommand


""" Agent """
agent = {
    'dH': 0,
    'frozen_steps': 0,
    'frozen_state': np.array([]),
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
}

""" AgentROS """
agent_ros = {
    # TODO: It might be worth putting this in json/yaml so C++ can read it.
    'trial_command_topic': 'gps/trial_command',
    'reset_command_topic': 'gps/reset_command',
    'sample_result_topic': 'gps/sample_result',
    'relax_command_topic': 'gps/relax_command',
    'data_command_topic': 'gps/data_command',
    'data_result_topic': 'gps/data_result',
    'trial_arm': ARM_RIGHT,
    'auxiliary_arm': ARM_LEFT,
    'reset_conditions': []  # Defines reset modes + positions for trial and auxiliary arms
}

""" AgentMuJoCo """
agent_mujoco = deepcopy(agent)
agent_mujoco.update({
    'substeps': 1,
    'append_pts_vel': False,
    'append_prev_state': False,
    'smooth_noise': False,
    'smooth_noise_sigma': 2.0,
    'smooth_noise_renormalize': True,
    'extra_phi_mean': np.array([]),
    'extra_phi_var': np.array([]),
})
