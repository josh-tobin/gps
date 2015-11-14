import rospy
import roslib
roslib.load_manifest('gps_agent_pkg')
import gps_agent_pkg
from gps_agent_pkg.msg import PositionCommand, TrialCommand, ControllerParams, LinGaussParams, SampleResult
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
import numpy as np
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.agent.ros.ros_utils import policy_to_msg, msg_to_sample
from gps.proto.gps_pb2 import *

POS_COM_TOPIC = '/gps_controller_position_command'
TRIAL_COM_TOPIC = '/gps_controller_trial_command'
TEST_TOPIC = '/test_sub'
RESULT_TOPIC = '/gps_controller_report'

def listen(msg):
    print msg.__class__

def listen_report(msg):
    print msg.__class__
    sensor_data = msg.sensor_data
    import pdb; pdb.set_trace();

def get_lin_gauss_test(T=50):
    dX = 14
    x0 = np.zeros(dX)
    x0[0] = 1.0
    lgpol = init_pd({'init_var': 0.01}, x0, 7, 7, dX, T)
    print 'T:', lgpol.T
    print 'dX:', lgpol.dX
    #Conver lgpol to message
    noise = np.zeros((T, 7))
    controller_params = policy_to_msg(lgpol, noise)
    return controller_params

def main():
    rospy.init_node('issue_com')
    pub = rospy.Publisher(TRIAL_COM_TOPIC, TrialCommand, queue_size=10)
    test_pub = rospy.Publisher(TEST_TOPIC, Empty, queue_size=10)
    sub = rospy.Subscriber(POS_COM_TOPIC, TrialCommand, listen)
    sub2 = rospy.Subscriber(RESULT_TOPIC, SampleResult, listen_report)
    #sub = rospy.Subscriber('/joint_states', JointState, listen)

    tc = TrialCommand()
    T = 20
    tc.controller = get_lin_gauss_test(T=T)
    tc.T = T
    tc.frequency = 20.0
    # NOTE: ordering of datatypes in state is determined by the order here
    tc.state_datatypes = [JOINT_ANGLES, JOINT_VELOCITIES]
    tc.obs_datatypes = tc.state_datatypes
    tc.ee_points = [0.0, 0.0, 0.0, 1.0,0.5,0.5]

    r = rospy.Rate(1)
    #while not rospy.is_shutdown():
    #    pub.publish(pc)
    #    r.sleep()
    #    print 'published!'
    r.sleep()
    test_pub.publish(Empty())
    pub.publish(tc)
    rospy.spin()

print "Testing"
if __name__ == "__main__":
    main()
