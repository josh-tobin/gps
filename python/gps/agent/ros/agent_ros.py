from copy import deepcopy
import rospy

from agent.agent import Agent
from agent.config import agent_ros
from ros_utils import ServiceEmulator, construct_sample_from_ros_msg, policy_object_to_ros_msg
from gps_agent_pkg.msg import TrialCommand, ControllerParams, SampleResult, PositionCommand, RelaxCommand, \
    DataRequest
from std_msgs.msg import Empty

ARM_LEFT = RelaxCommand.LEFT_ARM
ARM_RIGHT = RelaxCommand.RIGHT_ARM


class AgentROS(Agent):
    """
    All communication between the algorithms and ROS is done through this class.
    """
    def __init__(self, hyperparams, sample_data, state_assembler):
        config = deepcopy(agent_ros)
        config.update(hyperparams)
        Agent.__init__(self, config, sample_data, state_assembler)
        self._init_pubs_and_subs()

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(self._hyperparams('trial_command_topic'), TrialCommand,
                                              self._hyperparams('trial_result_topic'), SampleResult)
        self._reset_service = ServiceEmulator(self._hyperparams('reset_command_topic'), PositionCommand,
                                              self._hyperparams('reset_result_topic'), SampleResult)
        self._relax_service = ServiceEmulator(self._hyperparams('relax_command_topic'), RelaxCommand,
                                              self._hyperparams('relax_result_topic'), Empty)
        self._data_service = ServiceEmulator(self._hyperparams('data_command_topic'), RelaxCommand,
                                             self._hyperparams('data_result_topic'), Empty)

    def get_data(self, data_type):
        """
        Request for the most recent value for data/sensor readings.
        Ex. Joint angles, end effector position, jacobians.

        Args:
            data_type: Integer code for a data type.
                These are defined in sample_data.gps_sample_types
        """
        msg = DataRequest()
        msg.data_type = data_type
        msg.stamp = rospy.get_rostime()
        result_msg = self._data_service.publish_and_wait(msg)
        assert result_msg.data_type == data_type
        return result_msg.data

    def relax_arm(self, arm):
        """
        Relax one of the arms of the robot.

        Args:
            arm: Either ARM_LEFT or ARM_RIGHT
        """
        relax_command = RelaxCommand()
        relax_command.stamp = rospy.get_rostime()
        relax_command.arm = arm
        self._relax_service.publish_and_wait(relax_command)

    def reset_arm(self, arm, mode, data):
        """
        Issues a position command to an arm
        Args:
            arm: Either 'trial_arm', or 'auxillary_arm'.
            mode: An integer code (defined in PositionCommand.msg)
            data: An array of floats.
        """
        reset_command = PositionCommand()
        reset_command.mode = mode
        reset_data = data
        reset_command.data = reset_data
        reset_command.stamp = rospy.get_rostime()
        reset_command.arm = self._hyperparams[arm]
        reset_sample = self._reset_service.publish_and_wait(reset_command)
        # TODO: Maybe verify that you reset to the correct position.

    def reset(self, condition):
        """
        Reset the agent to be ready for a particular experiment condition.

        Args:
            condition (int): An index into hyperparams['reset_conditions']
        """
        condition_data = self._hyperparams['reset_conditions'][condition]
        self.reset_arm('trial_arm',
                       condition_data['trial_arm']['mode'],
                       condition_data['trial_arm']['data'])
        self.reset_arm('auxillary_arm',
                       condition_data['auxillary_arm']['mode'],
                       condition_data['auxillary_arm']['data'])

    def sample(self, policy, T):
        """
        Execute a policy and collect a sample

        Args:
            policy: A Policy object (ex. LinGauss, or CaffeNetwork)
            T: Trajectory length

        Returns:
            A Sample object
        """
        # Execute Trial
        trial_command = TrialCommand()
        trial_command.policy = policy_object_to_ros_msg(policy)
        trial_command.stamp = rospy.get_rostime()
        trial_command.T = T
        trial_command.frequency = self._hyperparams['frequency']
        sample_msg = self._trial_service.publish_and_wait(trial_command)

        sample = construct_sample_from_ros_msg(sample_msg)
        return sample
