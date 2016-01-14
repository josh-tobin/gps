import rospy
import numpy as np

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps_agent_pkg.msg import ControllerParams, LinGaussParams
from gps.agent.agent_utils import generate_noise
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import LIN_GAUSS_CONTROLLER


def msg_to_sample(ros_msg, agent):
    """
    Convert a SampleResult ROS message into a Sample python object.
    """
    sample = Sample(agent)
    for sensor in ros_msg.sensor_data:
        sensor_id = sensor.data_type
        shape = np.array(sensor.shape)
        data = np.array(sensor.data).reshape(shape)
        sample.set(sensor_id, data)
    return sample


def policy_to_msg(policy, noise):
    """
    Convert a policy object to a ROS ControllerParams message.
    """
    msg = ControllerParams()
    if isinstance(policy, LinearGaussianPolicy):
        msg.controller_to_execute = LIN_GAUSS_CONTROLLER
        msg.lingauss = LinGaussParams()
        msg.lingauss.T = policy.T
        msg.lingauss.dX = policy.dX
        msg.lingauss.dU = policy.dU
        msg.lingauss.K_t = policy.K.reshape(policy.T*policy.dX*policy.dU).tolist()
        msg.lingauss.k_t = policy.fold_k(noise).reshape(policy.T*policy.dU).tolist()
    else:
        raise NotImplementedError("Unknown policy object: %s" % policy)
    return msg


class TimeoutException(Exception):
    """
    Exception thrown on timeouts.
    """
    def __init__(self, sec_waited):
        super(TimeoutException, self).__init__("Timed out after %f seconds", sec_waited)


class ServiceEmulator(object):
    """
    Emulates a ROS service (request-response) from a publisher-subscriber pair.
    Args:
        pub_topic: Publisher topic.
        pub_type: Publisher message type.
        sub_topic: Subscriber topic.
        sub_type: Subscriber message type.
    """
    def __init__(self, pub_topic, pub_type, sub_topic, sub_type):
        self._pub = rospy.Publisher(pub_topic, pub_type)
        self._sub = rospy.Subscriber(sub_topic, sub_type, self._callback)

        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg):
        """
        Publish a message without waiting for response.
        """
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg, timeout=5.0, poll_delay=0.01, check_id=False):
        """
        Publish a message and wait for the response.
        Args:
            pub_msg: Message to publish.
            timeout: Timeout in seconds.
            poll_delay: Speed of polling for the subscriber message in seconds.
            check_id: If enabled, will only return messages with a matching id field.
        Returns:
            sub_msg: Subscriber message.
        """
        if check_id:  # This is not yet implemented in C++.
            raise NotImplementedError()

        self._waiting = True
        self.publish(pub_msg)

        time_waited = 0
        while self._waiting:
            rospy.sleep(poll_delay)
            time_waited += 0.01
            if time_waited > timeout:
                raise TimeoutException(time_waited)
        return self._subscriber_msg
