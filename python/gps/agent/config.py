"""Default configuration and hyperparameter values for Agent objects

"""
from copy import deepcopy
import numpy as np

#TODO: commenting out AgentROS stuff for now because the following import
#      seems to be breaking things. will have to revisit this
#from gps.agent.ros.agent_ros import ARM_RIGHT, ARM_LEFT
#from gps_agent_pkg.msg import PositionCommand


""" Agent """
agent = {
    'dH': 0,
    'x0var': 0,
    'noisy_body_idx': np.array([]),
    'noisy_body_var': np.array([]),
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'smooth_noise': False,
    'smooth_noise_var': 2.0,
    'smooth_noise_renormalize': True,
}

try:
    import roslib
    roslib.load_manifest('gps_agent_pkg')
    from gps_agent_pkg.msg import PositionCommand
    """ AgentROS """
    agent_ros = {
        # TODO: It might be worth putting this in json/yaml so C++ can read it.
        'trial_command_topic': 'gps_controller_trial_command',
        'reset_command_topic': 'gps_controller_position_command',
        'relax_command_topic': 'gps_controller_relax_command',
        'data_command_topic': 'gps_controller_data_command',
        'sample_result_topic': 'gps_controller_report',
        'trial_arm': PositionCommand.RIGHT_ARM,
        'auxiliary_arm': PositionCommand.LEFT_ARM,
        'trial_timeout': 20,  # Give this many seconds to execute a trial.
        'reset_conditions': [],  # Defines reset modes + positions for trial and auxiliary arms
        'frequency': 20,
        'smooth_noise': False,
        'smooth_noise_var': 2.0,
        'smooth_noise_renormalize': True,
    }
except ImportError as e:
    print 'No ROS enabled', e

""" AgentMuJoCo """
agent_mujoco = deepcopy(agent)
agent_mujoco.update({
    'substeps': 1,
})
