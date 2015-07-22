"""Default configuration and hyperparameter values for Agent objects

"""
import numpy as np

from copy import deepcopy

#TODO: commenting out AgentROS stuff for now because the following import
#      seems to be breaking things. will have to revisit this
#from agent.ros.agent_ros import ARM_RIGHT, ARM_LEFT
#from gps_agent_pkg.msg import PositionCommand


""" Agent """
agent = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': True,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': True,
}

#""" AgentROS """
#agent_ros = {
#    # TODO: It might be worth putting this in json/yaml so C++ can read it.
#    'trial_command_topic': 'gps/trial_command',
#    'reset_command_topic': 'gps/reset_command',
#    'relax_command_topic': 'gps/relax_command',
#    'data_command_topic': 'gps/data_command',
#    'sample_result_topic': 'gps/sample_result',
#    'trial_arm': ARM_RIGHT,
#    'auxiliary_arm': ARM_LEFT,
#    'trial_timeout': 20,  # Give this many seconds to execute a trial.
#    'reset_conditions': []  # Defines reset modes + positions for trial and auxiliary arms
#}

""" AgentMuJoCo """
agent_mujoco = deepcopy(agent)
agent_mujoco.update({
    'substeps': 1,
})
