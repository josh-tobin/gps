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

POS_COM_TOPIC = '/gps_controller_position_command'
TRIAL_COM_TOPIC = '/gps_controller_trial_command'
TEST_TOPIC = '/test_sub'

def listen(msg):
    print msg.__class__

def get_lin_gauss_test(T=50):
    dX = 14
    lgpol = init_pd({}, np.zeros(dX), 7, 7, dX, T)
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
    tc.controller = get_lin_gauss_test()
    tc.T = 50
    tc.frequency = 20
    tc.state_datatypes = [JOINT_STATES, JOINT_VELOCITIES]
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
    get_lin_gauss_test()
