import rospy

def construct_sample_from_ros_msg(ros_msg):
    raise NotImplementedError()


def policy_object_to_ros_msg(policy_object):
    raise NotImplementedError()


class TimeoutException(Exception):
    """ Exception thrown on timeouts """
    def __init__(self, sec_waited):
        super(TimeoutException, self).__init__("Timed out after %f seconds", sec_waited)


class ServiceEmulator(object):
    """
    Emulates a ROS service (request-response) from a publisher-subscriber pair.

    Args:
        pub_topic (string): Publisher topic
        pub_type (class): Publisher message type
        sub_topic (string): Subscriber topic
        sub_type (class): Subscriber message type
        timeout (float, optional): Timeout in seconds. Default 5
    """
    def __init__(self, pub_topic, pub_type,
                        sub_topic, sub_type, timeout=5.0):
        self._pub = rospy.Publisher(pub_topic, pub_type)
        self._sub = rospy.Subscriber(sub_topic, sub_type, self._callback)

        self.timeout = timeout
        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg):
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg):
        """
        Publish a message and wait for the response.

        Args:
            pub_msg: Message to publish
        Returns:
            sub_msg: Subscriber message
        """
        self._waiting = True
        self.publish(pub_msg)

        time_waited = 0
        while self._waiting:
            rospy.sleep(0.01)
            time_waited += 0.01
            if time_waited > self.timeout:
                raise TimeoutException(time_waited)
        return self._subscriber_msg