""" Default configuration and hyperparameters for agent objects. """
import numpy as np


# Agent
AGENT = {
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

    # AgentROS
    AGENT_ROS = {
        #TODO: It might be worth putting this in JSON/yaml format so C++
        #      can read it.
        'trial_command_topic': 'gps_controller_trial_command',
        'reset_command_topic': 'gps_controller_position_command',
        'relax_command_topic': 'gps_controller_relax_command',
        'data_request_topic': 'gps_controller_data_request',
        'sample_result_topic': 'gps_controller_report',
        'trial_timeout': 20,  # Give this many seconds for a trial.
        'reset_conditions': [],  # Defines reset modes + positions for
                                 # trial and auxiliary arms.
        'frequency': 20,
        'end_effector_points': np.array([]),
        #TODO: Actually pass in low gains and high gains and use both
        #      for the position controller.
        'pid_params': np.array([
            1200, 0.0, 30, 4,
            1200, 0.0, 10, 4,
            1000, 0.0, 6, 4,
            700, 0.0, 4, 4,
            300, 0.0, 6, 2,
            300, 0.0, 4, 2,
            300, 0.0, 4, 2
        ]),
    }
except ImportError as e:
    LOGGER.debug('No ROS enabled: %s', e)
except rospkg.common.ResourceNotFound as e:
    LOGGER.debug('No gps_agent_pkg: %s', e)


# AgentMuJoCo
AGENT_MUJOCO = {
    'substeps': 1,
}
