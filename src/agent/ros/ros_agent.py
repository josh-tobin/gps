import rospy

from agent import Agent
from ros_utils import ServiceEmulator, construct_sample_from_ros_msg
from gps_agent_pkg.msg import TrialCommand, ControllerParams, SampleResult


class ROSAgent(Agent):
    """
    """
    def __init__(self, hyperparams, common_hyperparams, sample_data, state_assembler):
        super(ROSAgent, self).__init__(hyperparams, common_hyperparams, sample_data, state_assembler)
        self._init_pubs_and_subs()

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator('pub_topic', TrialCommand, 'sub_topic', SampleResult)

        self._reset_publisher = None
        self._relax_publisher = None


    def sample(self, policy, T):
        """
        Execute a policy and collect a sample

        Args:
            policy: A Policy object (ex. LinGauss, or CaffeNetwork)
            T: Trajectory length

        Returns:
            A Sample object
        """
        trial_command = TrialCommand()
        trial_command.policy = policy.to_ros_msg()
        trial_command.stamp = rospy.get_rostime()
        trial_command.T = T
        trial_command.frequency = self._hyperparams['frequency']

        sample_msg = self._trial_service.publish_and_wait(trial_command)
        sample = construct_sample_from_ros_msg(sample_msg)

        return sample
