import time
import numpy as np
from sensor_msgs.msg import JointState
import rospy
import logging

from matlab_interface import get_controller

logging.basicConfig(level=logging.DEBUG)

def main():
	rospy.init_node('mpc_node')
	controller = get_controller('/home/justin/RobotRL/test/onlinecont.mat')
	def state_callback(msg):
		start_time = time.time()

		X = np.array(msg.position)
		t = int(msg.velocity[0])
		if t == 98:
			print 'Saving outputs!'
		obs = None
		noise = None

		action = controller.act(X, obs, t, noise)
		print action
		#action.fill(0.0)

		response = JointState()
		response.velocity = (t+1,)
		response.effort = action
		#elapsed_time = time.time()-start_time
		#time.sleep(0.045-elapsed_time)			
		action_publisher.publish(response)
		elapsed_time = time.time()-start_time
		print 'Calculation took %f s for T=%d' % (elapsed_time, t)

	action_publisher = rospy.Publisher('/ddp/mat_controller_action', JointState)
	state_subscriber = rospy.Subscriber('/ddp/mat_controller_state', JointState, state_callback)

	print 'Python controller initialized and spinning...'
	rospy.spin()

if __name__ == "__main__":
	main()