import rospy

def construct_sample_from_ros_msg(ros_msg):
    pass

class ServiceEmulator(object):
    """
    Emulates a ROS service from a publisher-subscriber pair.
    """
    def __init__(self, pub_topic, pub_type,
                        sub_topic, sub_type):
        self._pub = rospy.Publisher(pub_topic, pub_type)
        self._sub = rospy.Subscriber(sub_topic, sub_type, self._callback)

        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg):
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg):
        self._waiting = True
        self.publish(pub_msg)
        while self._waiting:
            rospy.sleep(0.01)
        return self._subscriber_msg