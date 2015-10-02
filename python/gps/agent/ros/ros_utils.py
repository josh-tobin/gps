import rospy
from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps_agent_pkg.msg import ControllerParams, LinGaussParams
from agent.agent_utils import generate_noise


def construct_sample_from_msg(ros_msg):
    """
    Convert a SampleResult ROS message into a Sample python object
    """
    raise NotImplementedError()


def policy_to_msg(policy):
    """
    Convert a policy object to a ROS ControllerParams message
    """
    msg = ControllerParams()
    if isinstance(policy, LinearGaussianPolicy):
        msg.controller_to_execute = ControllerParams.LIN_GAUSS_CONTROLLER
        msg.lingauss = LinGaussParams()
        msg.lingauss.K_t = policy.K
         #TODO: Initialize noise somewhere else
        noise = generate_noise(policy.T, policy.dU)
        msg.lingauss.k_t = policy.fold_k(noise)
    else:
        raise NotImplementedError("Unknown policy object: %s" % policy)
    return msg


class TimeoutException(Exception):
    """ Exception thrown on timeouts """
    def __init__(self, sec_waited):
        super(TimeoutException, self).__init__("Timed out after %f seconds", sec_waited)


class ServiceEmulator(object):
    """
    Emulates a ROS service (request-response) from a publisher-subscriber pair.

    Args:
        pub_topic (string): Publisher topic
        pub_type (class): Publisher message type. Must have a header field (for seq id checking)
        sub_topic (string): Subscriber topic
        sub_type (class): Subscriber message type. Must have a header field (for seq id checking)
    """
    def __init__(self, pub_topic, pub_type, sub_topic, sub_type):
        self._pub = rospy.Publisher(pub_topic, pub_type)
        self._sub = rospy.Subscriber(sub_topic, sub_type, self._callback)

        self._waiting = False
        self._header_seq = -1
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            if message.header.seq != self._header_seq:
                return
            self._subscriber_msg = message
            self._waiting = False
            self._header_seq = -1

    def publish(self, pub_msg):
        """ Publish a message without waiting for response """
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg, timeout=5.0):
        """
        Publish a message and wait for the response.

        Args:
            pub_msg (pub_type): Message to publish
            timeout (float, optional): Timeout in seconds. Default 5.0
        Returns:
            sub_msg (sub_type): Subscriber message
        """
        self._waiting = True
        self._header_seq = pub_msg.header.seq
        self.publish(pub_msg)

        time_waited = 0
        while self._waiting:
            rospy.sleep(0.01)
            time_waited += 0.01
            if time_waited > timeout:
                raise TimeoutException(time_waited)
        return self._subscriber_msg