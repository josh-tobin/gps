import time
import numpy as np
from sensor_msgs.msg import JointState
import rospy

from matlab_interface import get_controller

def main():
	rospy.init_node('mpc_node')
	controller = get_controller('/home/justin/RobotRL/test/onlinecont.mat')
	def state_callback(msg):
		start_time = time.time()

		X = np.array(msg.position)
		t = int(msg.velocity[0])
		obs = None
		noise = None

		action = controller.act(X, obs, t, noise)

		response = JointState()
		response.velocity = (t+1,)
		response.effort = action
		action_publisher.publish(response)
		elapsed_time = time.time()-start_time
		print 'Calculation took %f s for T=%d' % (elapsed_time, t)

	action_publisher = rospy.Publisher('/ddp/mat_controller_action', JointState)
	state_subscriber = rospy.Subscriber('/ddp/mat_controller_state', JointState, state_callback)

	print 'Python controller initialized and spinning...'
	rospy.spin()

if __name__ == "__main__":
	main()