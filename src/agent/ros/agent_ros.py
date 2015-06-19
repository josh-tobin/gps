import rospy

from agent import Agent
from ros_utils import ServiceEmulator, construct_sample_from_ros_msg, policy_object_to_ros_msg
from gps_agent_pkg.msg import TrialCommand, ControllerParams, SampleResult, PositionCommand, RelaxCommand


class AgentROS(Agent):
    """
    """
    def __init__(self, hyperparams, common_hyperparams, sample_data, state_assembler):
        super(AgentROS, self).__init__(hyperparams, common_hyperparams, sample_data, state_assembler)
        self._init_pubs_and_subs()

    def _init_pubs_and_subs(self):
        #TODO: Read topics from a yaml/config file
        self._trial_service = ServiceEmulator('pub_topic', TrialCommand, 'sub_topic', SampleResult)
        self._reset_service = ServiceEmulator('pub_topic', PositionCommand, 'sub_topic', SampleResult)
        self._relax_publisher = rospy.Publisher('relax_topic', RelaxCommand)

    def relax_arm(self, arm):
        raise NotImplementedError()

    def sample(self, policy, T):
        """
        Execute a policy and collect a sample

        Args:
            policy: A Policy object (ex. LinGauss, or CaffeNetwork)
            T: Trajectory length

        Returns:
            A Sample object
        """
        # Reset robot
        reset_position = None #TODO - read from hyperparams?
        reset_command = PositionCommand()
        reset_command.data = reset_position
        reset_command.stamp = rospy.get_rostime()
        reset_command.arm = None #TODO
        reset_command.mode = None #TODO
        reset_sample = self._reset_service.publish_and_wait(reset_command)
        # TODO: Maybe verify that you reset to the correct position.

        # Execute Trial
        trial_command = TrialCommand()
        trial_command.policy = policy_object_to_ros_msg(policy)
        trial_command.stamp = rospy.get_rostime()
        trial_command.T = T
        trial_command.frequency = self._hyperparams['frequency']
        sample_msg = self._trial_service.publish_and_wait(trial_command)

        sample = construct_sample_from_ros_msg(sample_msg)
        return sample
