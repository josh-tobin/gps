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
from visualization_msgs.msg import Marker
from geometry_msgs.msg import WrenchStamped, Wrench, Twist, Vector3, Point

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)

BASE_LINK = '/torso_lift_link'

def pub_viz_vec(viz_pub, points_to_viz, color=[1.0,1.0,1.0], id=0, base_link='/torso_lift_link', point_type=Marker.LINE_STRIP):
    points = Marker()
    points.header.frame_id = base_link
    points.id=id
    points.type = point_type
    for point in points_to_viz:
        pp = Point()
        pp.x = point[0]; pp.y = point[1]; pp.z = point[2]
        points.points.append(pp)
    points.color.r = color[0]
    points.color.g = color[1]
    points.color.b = color[2]
    points.color.a = 1.0 
    points.scale.x = 0.01
    points.scale.y = 0.01
    points.scale.z = 1.0
    points.lifetime = rospy.Duration(1)  # 1 Second
    viz_pub.publish(points)

def visualize_forward_pass(fwd, viz_pub, color=[0.0,1.0,1.0], id=0):
    H, Dee = fwd.shape
    assert Dee == 3
    pnts = [fwd[t,:] for t in range(H)] 
    pub_viz_vec(viz_pub, pnts, color=color, id=id)

def main():
    rospy.init_node('mpc_node')
    controller = get_controller('/home/justin/RobotRL/test/onlinecont.mat')
    def state_callback(msg):
        start_time = time.time()

        X = np.array(msg.X)
        t = msg.t

        if t==99:
            pass
        	#with open('plane_ma2.pkl', 'w') as f:
        	#	cPickle.dump(controller, f)

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

        fwd_ee = controller.get_forward_end_effector(0)
        if fwd_ee is not None:
            visualize_forward_pass(controller.get_forward_end_effector(0), visualization_pub, color=[1.0,0.0,0.0], id=0)
            visualize_forward_pass(controller.get_forward_end_effector(1), visualization_pub, color=[0.0,1.0,0.0], id=1)
            visualize_forward_pass(controller.get_forward_end_effector(2), visualization_pub, color=[0.0,0.0,1.0], id=2)
            eetgt = controller.cost.get_ee_tgt(99)
            pub_viz_vec(visualization_pub, [eetgt[0:3], eetgt[3:6], eetgt[6:9]], point_type=Marker.POINTS, id=3)
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
    visualization_pub = rospy.Publisher('/ddp/online_controller_viz', Marker)
    state_subscriber = rospy.Subscriber('/ddp/mat_controller_state', MPCState, state_callback)

    print 'Python controller initialized and spinning...'
    rospy.spin()

if __name__ == "__main__":
    main()
