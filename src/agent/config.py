"""Default configuration and hyperparameter values for Agent objects

"""
from agent.ros.agent_ros import ARM_RIGHT, ARM_LEFT
from gps_agent_pkg.msg import PositionCommand


""" AgentROS """
agent_ros = {
    # TODO: It might be worth putting this in json/yaml so C++ can read it.
    'trial_command_topic': 'gps/trial_command',
    'trial_result_topic': 'gps/trial_result',
    'reset_command_topic': 'gps/reset_command',
    'reset_result_topic': 'gps/reset_result',
    'relax_command_topic': 'gps/relax_command',
    'relax_result_topic': 'gps/relax_result',
    'data_command_topic': 'gps/data_command',
    'data_result_topic': 'gps/data_result',
    'trial_arm': ARM_RIGHT,
    'auxillary_arm': ARM_LEFT,
    'reset_conditions': []  # Defines reset modes + positions for trial and auxillary arms
}

""" AgentMuJoCo """
agent_mujoco = {
}