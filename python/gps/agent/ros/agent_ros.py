from copy import deepcopy
import rospy

from gps.agent.agent import Agent
from gps.agent.config import agent_ros
from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, policy_to_msg
from gps_agent_pkg.msg import TrialCommand, ControllerParams, SampleResult, PositionCommand, RelaxCommand, \
    DataRequest
from std_msgs.msg import Empty

#from gps.agent.ros.ros_utils import ServiceEmulator, construct_sample_from_ros_msg, policy_object_to_ros_msg

ARM_LEFT = RelaxCommand.LEFT_ARM
ARM_RIGHT = RelaxCommand.RIGHT_ARM


class AgentROS(Agent):
    """
    All communication between the algorithms and ROS is done through this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(agent_ros)
        config.update(hyperparams)
        Agent.__init__(self, config)
        rospy.init_node('gps_agent_ros_node')
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(self._hyperparams['trial_command_topic'], TrialCommand,
                                              self._hyperparams['sample_result_topic'], SampleResult)
        self._reset_service = ServiceEmulator(self._hyperparams['reset_command_topic'], PositionCommand,
                                              self._hyperparams['sample_result_topic'], SampleResult)
        self._relax_service = ServiceEmulator(self._hyperparams['relax_command_topic'], RelaxCommand,
                                              self._hyperparams['sample_result_topic'], SampleResult)
        self._data_service = ServiceEmulator(self._hyperparams['data_command_topic'], RelaxCommand,
                                             self._hyperparams['sample_result_topic'], SampleResult)

    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2**32)  # Max uint32
        return self._seq_id

    def get_data(self, data_type, arm=-1):
        """
        Request for the most recent value for data/sensor readings.
        Ex. Joint angles, end effector position, jacobians.

        Args:
            data_type: Integer code for a data type.
                These are defined in proto.gps_pb2
            arm: Either ARM_LEFT or ARM_RIGHT, or -1 if not applicable
        """
        msg = DataRequest()
        msg.header.seq = self._get_next_seq_id()
        msg.header.stamp = rospy.get_rostime()
        msg.data_type = data_type
        msg.arm = arm
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
        relax_command.header.seq = self._get_next_seq_id()
        relax_command.header.stamp = rospy.get_rostime()
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
        reset_command.arm = self._hyperparams[arm]
        reset_sample = self._reset_service.publish_and_wait(reset_command)
        # TODO: Maybe verify that you reset to the correct position.

    def reset(self, condition):
        """
        Reset the agent to be ready for a particular experiment condition.

        Args:
            condition (int): An index into hyperparams['reset_conditions']
        """
        #TODO: Discuss reset + implement reset
        condition_data = self._hyperparams['reset_conditions'][condition]
        self.reset_arm('trial_arm',
                       condition_data['trial_arm']['mode'],
                       condition_data['trial_arm']['data'])
        self.reset_arm('auxiliary_arm',
                       condition_data['auxiliary_arm']['mode'],
                       condition_data['auxiliary_arm']['data'])

    #def sample(self, policy, T, condition=0):
    def sample(self, policy, condition):
        """
        Execute a policy and collect a sample

        Args:
            policy: A Policy object (ex. LinGauss, or CaffeNetwork)
            T: Trajectory length
            condition (int): Which condition setup to run.

        Returns:
            A Sample object
        """

        # Execute Trial
        trial_command = TrialCommand()
        trial_command.controller = policy_to_msg(policy)
        #trial_command.policy = policy_object_to_ros_msg(policy)
        trial_command.header.seq = self._get_next_seq_id()
        trial_command.header.stamp = rospy.get_rostime()
        trial_command.T = self.T
        trial_command.frequency = self._hyperparams['frequency']
        #TODO: Read this from hyperparams['state_include']
        print 'TODO: Using JOINT_ANGLES, JOINT_VELOCITIES as state (see agent_ros.py)'

        from proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES
        trial_command.state_datatypes = [JOINT_ANGLES, JOINT_VELOCITIES]
        trial_command.obs_datatypes = [JOINT_ANGLES, JOINT_VELOCITIES]
        sample_msg = self._trial_service.publish_and_wait(trial_command, timeout=self._hyperparams['trial_timeout'])

        sample = msg_to_sample(sample_msg)
        #sample = construct_sample_from_ros_msg(sample_msg)
        self._samples[condition].append(sample)
        #return sample
