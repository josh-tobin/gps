"""
Default configuration and hyperparameter values for Agent objects.
"""
import numpy as np

from gps.proto.gps_pb2 import AUXILIARY_ARM, TRIAL_ARM


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


try:
    import roslib
    import rospkg
    roslib.load_manifest('gps_agent_pkg')
    from gps_agent_pkg.msg import PositionCommand
    """ AgentROS """
    agent_ros = {
        #TODO: It might be worth putting this in json/yaml so C++ can read it.
        'trial_command_topic': 'gps_controller_trial_command',
        'reset_command_topic': 'gps_controller_position_command',
        'relax_command_topic': 'gps_controller_relax_command',
        'data_request_topic': 'gps_controller_data_request',
        'sample_result_topic': 'gps_controller_report',
        'trial_timeout': 20,  # Give this many seconds to execute a trial.
        'reset_conditions': [],  # Defines reset modes + positions for trial and auxiliary arms
        'frequency': 20,
        'end_effector_points': np.array([]),
        #TODO: Actually pass in low gains and high gains and use both for the position controller.
        'pid_params': np.array([
            60, 200, 10, 4,
            600, 500, 10, 4,
            1000, 600, 6, 4,
            700, 450, 4, 4,
            300, 300, 6, 2,
            300, 300, 4, 2,
            300, 300, 4, 2
        ]),
    }
except ImportError as e:
    print 'No ROS enabled', e
except rospkg.common.ResourceNotFound as e:
    print 'No gps_agent_pkg', e


""" AgentMuJoCo """
agent_mujoco = {
    'substeps': 1,
}
