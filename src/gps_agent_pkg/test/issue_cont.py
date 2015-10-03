import rospy
import roslib
roslib.load_manifest('gps_agent_pkg')
import gps_agent_pkg
from gps_agent_pkg.msg import PositionCommand, TrialCommand, ControllerParams, LinGaussParams
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
import numpy as np
from algorithm.policy.lin_gauss_init import init_pd
from agent.ros.ros_utils import policy_to_msg
from proto.gps_pb2 import *

POS_COM_TOPIC = '/gps_controller_position_command'
TRIAL_COM_TOPIC = '/gps_controller_trial_command'
TEST_TOPIC = '/test_sub'

def listen(msg):
    print msg.__class__

def get_lin_gauss_test(T=50):
    dX = 14
    x0 = np.zeros(dX)
    x0[0] = 1.0
    lgpol = init_pd({}, x0, 7, 7, dX, T)
    print 'T:', lgpol.T
    print 'dX:', lgpol.dX
    #Conver lgpol to message
    controller_params = policy_to_msg(lgpol)
    return controller_params

def main():
    rospy.init_node('issue_com')
    pub = rospy.Publisher(TRIAL_COM_TOPIC, TrialCommand, queue_size=10)
    test_pub = rospy.Publisher(TEST_TOPIC, Empty, queue_size=10)
    sub = rospy.Subscriber(POS_COM_TOPIC, TrialCommand, listen)
    #sub = rospy.Subscriber('/joint_states', JointState, listen)

    tc = TrialCommand()
    T = 100
    tc.controller = get_lin_gauss_test(T=T)
    tc.T = T
    tc.frequency = 20.0
    # NOTE: ordering of datatypes in state is determined by the order here
    tc.state_datatypes = [JOINT_ANGLES, JOINT_VELOCITIES]
    tc.obs_datatypes = tc.state_datatypes

    r = rospy.Rate(1)
    #while not rospy.is_shutdown():
    #    pub.publish(pc)
    #    r.sleep()
    #    print 'published!'
    r.sleep()
    test_pub.publish(Empty())
    pub.publish(tc)

print "Testing"
if __name__ == "__main__":
    main()
