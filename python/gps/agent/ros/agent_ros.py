from copy import deepcopy
import rospy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import agent_ros
from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, policy_to_msg
from gps_agent_pkg.msg import TrialCommand, ControllerParams, SampleResult, PositionCommand, RelaxCommand, \
    DataRequest
from std_msgs.msg import Empty
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM



class AgentROS(Agent):
    """
    All communication between the algorithms and ROS is done through this class.
    """
    def __init__(self, hyperparams, init_node=True):
        """
        Initialize agent.

        Args:
            hyperparams: dictionary of hyperparameters
            init_node: whether or not to initialize a new ros node.
        """
        config = deepcopy(agent_ros)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_ros_node')
        self._init_pubs_and_subs()
        self._seq_id = 0  # Used for setting seq in ROS commands

        conditions = self._hyperparams['conditions']

        def setup(value, n):
            if not isinstance(value, list):
                try:
                    return [value.copy() for _ in range(n)]
                except AttributeError:
                    return [value for _ in range(n)]
            assert len(value) == n, 'number of elements must match number of conditions or be 1'
            return value

        self.x0 = []
        for field in ('x0', 'ee_points_tgt', 'reset_conditions'):
            self._hyperparams[field] = setup(self._hyperparams[field], conditions)
        self.x0 = self._hyperparams['x0']

        r = rospy.Rate(1)
        r.sleep()

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(self._hyperparams['trial_command_topic'], TrialCommand,
                                              self._hyperparams['sample_result_topic'], SampleResult)
        self._reset_service = ServiceEmulator(self._hyperparams['reset_command_topic'], PositionCommand,
                                              self._hyperparams['sample_result_topic'], SampleResult)
        self._relax_service = ServiceEmulator(self._hyperparams['relax_command_topic'], RelaxCommand,
                                              self._hyperparams['sample_result_topic'], SampleResult)
        self._data_service = ServiceEmulator(self._hyperparams['data_request_topic'], DataRequest,
                                             self._hyperparams['sample_result_topic'], SampleResult)

    def _get_next_seq_id(self):
        self._seq_id = (self._seq_id + 1) % (2**32)  # Max uint32
        return self._seq_id

    def get_data(self, arm=TRIAL_ARM):
        """
        Request for the most recent value for data/sensor readings.
        Ex. Joint angles, end effector position, jacobians.
        Returns entire sample report (all available data) in sample object

        Args:
            arm: TRIAL_ARM or AUXILIARY_ARM
        """
        request = DataRequest()
        request.id = self._get_next_seq_id()
        request.arm = arm
        request.stamp = rospy.get_rostime()
        result_msg = self._data_service.publish_and_wait(request)
        # TODO - make IDs match, assert that they match elsewhere here.
        #assert result_msg.id == request.id
        sample = msg_to_sample(result_msg, self)
        return sample

    # TODO - the following could be more general by being relax_actuator and reset_actuator
    def relax_arm(self, arm):
        """
        Relax one of the arms of the robot.

        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM
        """
        relax_command = RelaxCommand()
        relax_command.id = self._get_next_seq_id()
        relax_command.stamp = rospy.get_rostime()
        relax_command.arm = arm
        self._relax_service.publish_and_wait(relax_command)

    def reset_arm(self, arm, mode, data):
        """
        Issues a position command to an arm
        Args:
            arm: Either TRIAL_ARM or AUXILIARY_ARM
            mode: An integer code (defined gps_pb2)
            data: An array of floats.
        """
        reset_command = PositionCommand()
        reset_command.mode = mode
        reset_command.data = data
        reset_command.pd_gains = self._hyperparams['pid_params']
        reset_command.arm = arm
        timeout = self._hyperparams['trial_timeout']
        reset_command.id = self._get_next_seq_id()
        reset_sample = self._reset_service.publish_and_wait(reset_command, \
            timeout=timeout)
        # TODO: Maybe verify that you reset to the correct position.

    def reset(self, condition):
        """
        Reset the agent to be ready for a particular experiment condition.

        Args:
            condition (int): An index into hyperparams['reset_conditions']
        """
        condition_data = self._hyperparams['reset_conditions'][condition]
        self.reset_arm(TRIAL_ARM,
                       condition_data[TRIAL_ARM]['mode'],
                       condition_data[TRIAL_ARM]['data'])
        self.reset_arm(AUXILIARY_ARM,
                       condition_data[AUXILIARY_ARM]['mode'],
                       condition_data[AUXILIARY_ARM]['data'])

    def sample(self, policy, condition, verbose=True):
        """
        Reset and execute a policy and collect a sample

        Args:
            policy: A Policy object (ex. LinGauss, or CaffeNetwork)
            condition (int): Which condition setup to run.

        Returns:
            A Sample object
        """
        self.reset(condition)

        #Generate noise
        noise = generate_noise(self.T, self.dU,
            smooth=self._hyperparams['smooth_noise'],
            var=self._hyperparams['smooth_noise_var'],
            renorm=self._hyperparams['smooth_noise_renormalize'])

        # Execute Trial
        trial_command = TrialCommand()
        trial_command.id = self._get_next_seq_id()
        trial_command.controller = policy_to_msg(policy, noise)
        trial_command.T = self.T
        trial_command.id = self._get_next_seq_id()
        trial_command.frequency = self._hyperparams['frequency']
        ee_points = self._hyperparams['end_effector_points']
        trial_command.ee_points = ee_points.reshape(ee_points.size).tolist()
        trial_command.ee_points_tgt = self._hyperparams['ee_points_tgt'][condition].tolist()
        trial_command.state_datatypes = self._hyperparams['state_include']
        trial_command.obs_datatypes = self._hyperparams['state_include']
        sample_msg = self._trial_service.publish_and_wait(trial_command, timeout=self._hyperparams['trial_timeout'])

        sample = msg_to_sample(sample_msg, self)
        self._samples[condition].append(sample)
        return sample
