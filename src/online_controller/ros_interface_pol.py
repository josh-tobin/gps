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

BASE_LINK = 'torso_lift_link'

def pub_viz_vec(viz_pub, ee_pos, v, color=[1.0,1.0,1.0], id=0, base_link='torso_lift_link'):
    ee_pnt = Point()
    ee_pnt.x = ee_pos[0]; ee_pnt.y = ee_pos[1]; ee_pnt.z=ee_pos[2]
        # Vector end point
    v_pnt = Point()
    v_pnt.x = ee_pos[0]+v[0]; v_pnt.y = ee_pos[1]+v[1]; v_pnt.z=ee_pos[2]+v[2]

    points = Marker()
    points.header.frame_id = base_link
    points.id=id
    points.type = Marker.LINE_LIST
    points.points.append(ee_pnt)
    points.points.append(v_pnt)
    points.color.r = color[0]
    points.color.g = color[1]
    points.color.b = color[2]
    points.color.a = 1.0 
    points.lifetime = rospy.Duration(1)  # 1 Second
    viz_pub.publish(points)

def solve_fk():
    rospy.wait_for_service('pr2_right_arm_kinematics/get_fk')
    rospy.wait_for_service('pr2_right_arm_kinematics/get_fk_solver_info')
    try:
        getfk_info = rospy.ServiceProxy('pr2_right_arm_kinematics/get_fk_solver_info', GetKinematicSolverInfo)
        fkinfo = getfk_info()
        print 'fkinfo:', fkinfo

        getfk = rospy.ServiceProxy('pr2_right_arm_kinematics/get_fk', GetPositionFK)
        request = GetPositionFK.Request()
        request.header.frame_id = BASE_LINK

        return resp1.sum
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def visualize_forward_pass(fwd):
    H, Djnt = fwd.shape
    assert Djnt == 7
    for t in range(H):
        jnts = fwd[t,:]

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

        # Set up forward pass visualization
        fwd_jnt_states = controller.get_forward_joint_states() # H x 7 matrix
        visualize_forward_pass(fwd_jnt_states)


    action_publisher = rospy.Publisher('/ddp/mat_controller_policy', LGPolicy)
    state_subscriber = rospy.Subscriber('/ddp/mat_controller_state', MPCState, state_callback)

    print 'Python controller initialized and spinning...'
    rospy.spin()

if __name__ == "__main__":
    main()
