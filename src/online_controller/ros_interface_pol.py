import time
import numpy as np
import roslib
roslib.load_manifest('ddp_controller_pkg')
from sensor_msgs.msg import JointState
from ddp_controller_pkg.msg import LGPolicy, MPCState
import rospy
import logging
import cPickle
from matlab_interface import get_controller

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)

def main():
    rospy.init_node('mpc_node')
    controller = get_controller('/home/justin/RobotRL/test/onlinecont.mat')
    def state_callback(msg):
        start_time = time.time()

        X = np.array(msg.X)
        t = msg.t

        #if t==98:
        #	with open('junk.pkl', 'w') as f:
        #		cPickle.dump(controller, f)

        obs = None
        noise = None
        dT = msg.dX+msg.dU
        empsig = np.array(msg.dynamics_sigma).reshape(dT+msg.dX, dT+msg.dX)
        empmu = np.array(msg.dynamics_mu).reshape(dT+msg.dX)
        prevx = np.array(msg.prevx).reshape(msg.dX)
        prevu = np.array(msg.prevu).reshape(msg.dU)

        #if t==1:
        #	import pdb; pdb.set_trace()
        lgpol = controller.act_pol(X, empmu, empsig, prevx, prevu, t)
        #action.fill(0.0)

        response = LGPolicy()
        response.dX = msg.dX
        response.dU = msg.dU
        response.t_start = t
        response.t_end = lgpol.T+t
        response.K_t = lgpol.K.flatten()
        response.k_t = lgpol.k.flatten()

        #elapsed_time = time.time()-start_time
        #time.sleep(0.045-elapsed_time)	

        action_publisher.publish(response)
        elapsed_time = time.time()-start_time
        print 'Calculation took %f s for T=%d' % (elapsed_time, t)

    action_publisher = rospy.Publisher('/ddp/mat_controller_policy', LGPolicy)
    state_subscriber = rospy.Subscriber('/ddp/mat_controller_state', MPCState, state_callback)

    print 'Python controller initialized and spinning...'
    rospy.spin()

if __name__ == "__main__":
    main()