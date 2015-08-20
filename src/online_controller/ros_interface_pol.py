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
    points.lifetime = rospy.Duration(60)  # 1 Second
    viz_pub.publish(points)

def visualize_forward_pass(fwd, viz_pub, color=[0.0,1.0,1.0], id=0):
    H, Dee = fwd.shape
    assert Dee == 3
    pnts = [fwd[t,:] for t in range(H)] 
    pub_viz_vec(viz_pub, pnts, color=color, id=id)

def visualize_dynamics(traj, dynamics, viz_pub, ee_idx=slice(14,23), nnet=None):
    traj_line = Marker()
    traj_line.header.frame_id = BASE_LINK
    traj_line.id=0
    traj_line.type = Marker.LINE_STRIP
    traj_line.color.r = 1.0
    traj_line.color.a = 1.0
    traj_line.scale.x = 0.01
    traj_line.scale.y = 0.01
    traj_line.scale.z = 0.1
    traj_line.lifetime = rospy.Duration(60)  # 1 Second

    dyn_line = Marker()
    dyn_line.header.frame_id = BASE_LINK
    dyn_line.id=1
    dyn_line.type = Marker.LINE_LIST
    dyn_line.color.b = 1.0
    dyn_line.color.a = 1.0
    dyn_line.scale.x = 0.01
    dyn_line.scale.y = 0.01
    dyn_line.scale.z = 0.1
    dyn_line.lifetime = rospy.Duration(60)  # 1 Second

    if nnet:
        nnet_line = Marker()
        nnet_line.header.frame_id = BASE_LINK
        nnet_line.id=999
        nnet_line.type = Marker.LINE_LIST
        nnet_line.color.g = 1.0
        nnet_line.color.a = 1.0
        nnet_line.scale.x = 0.01
        nnet_line.scale.y = 0.01
        nnet_line.scale.z = 0.1
        nnet_line.lifetime = rospy.Duration(60)  # 1 Second

    for t in range(1,len(traj)):
        if t%2 == 0: # Too dense
            pass
            #continue
        xu = traj[t]

        eept = xu[ee_idx]
        pp = Point()
        pp.x = eept[0]; pp.y = eept[1]; pp.z = eept[2]
        traj_line.points.append(pp)
        dyn_line.points.append(pp)
        if nnet:
            nnet_line.points.append(pp)

        F, f = dynamics[t]
        x_next = F.dot(xu) + f 
        eept = x_next[ee_idx]
        pp = Point(); pp.x = eept[0]; pp.y = eept[1]; pp.z = eept[2]
        dyn_line.points.append(pp)

        if nnet:
            x_next = nnet.fwd_single(xu.astype(np.float32))
            #print np.all(x_next[ee_idx] == eept)
            eept = x_next[ee_idx]
            pp = Point(); pp.x = eept[0]; pp.y = eept[1]; pp.z = eept[2]
            nnet_line.points.append(pp)


    viz_pub.publish(traj_line)
    viz_pub.publish(dyn_line)
    viz_pub.publish(nnet_line)

def process_jac(jac, eerot, ee_sites, dX):
    n_sites = ee_sites.shape[0]
    n_actuator = jac.shape[1]

    Jx = np.zeros((3*n_sites, dX))
    Jr = np.zeros((3*n_sites, dX))

    iq = slice(0,n_actuator)
    # Process each site.
    for i in range(n_sites):
        site_start = i*3
        site_end = (i+1)*3
        # Initialize.
        ovec = ee_sites[i]
        
        Jx[site_start:site_end, iq] = jac[0:3,:]
        Jr[site_start:site_end, iq] = jac[3:6,:]
        
        # Compute site Jacobian.
        ovec = eerot.dot(ovec)
        Jx[site_start:site_end, iq] += \
            np.c_[Jr[site_start+1, iq].dot(ovec[2]) - Jr[site_start+2, iq].dot(ovec[1]) , 
             Jr[site_start+2, iq].dot(ovec[0]) - Jr[site_start, iq].dot(ovec[2]) , 
             Jr[site_start, iq].dot(ovec[1]) - Jr[site_start+1, iq].dot(ovec[0])].T
    return Jx

def main():
    rospy.init_node('mpc_node')
    controller = get_controller('/home/justin/RobotRL/test/onlinecont.mat')
    def state_callback(msg):
        start_time = time.time()

        X = np.array(msg.X)
        t = msg.t


        obs = None
        noise = None
        dT = msg.dX+msg.dU
        empsig = np.array(msg.dynamics_sigma).reshape(dT+msg.dX, dT+msg.dX)
        empmu = np.array(msg.dynamics_mu).reshape(dT+msg.dX)
        prevx = np.array(msg.prevx).reshape(msg.dX)
        prevu = np.array(msg.prevu).reshape(msg.dU)
        eerot = np.array(msg.eerot).reshape(3,3).T
        eejac = np.array(msg.jacobian).reshape(msg.dU, 6).T
        sitejac = process_jac(eejac, eerot, controller.ee_sites, msg.dX)

        #if t==1:
        #	import pdb; pdb.set_trace()
        lgpol = controller.act_pol(X, empmu, empsig, prevx, prevu, sitejac, eejac, t)

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

        if t==controller.maxT-1:
            visualize_dynamics(controller.rviz_traj, controller.dyn_hist_list, visualization_pub, nnet=controller.dyn_net)

    action_publisher = rospy.Publisher('/ddp/mat_controller_policy', LGPolicy)
    visualization_pub = rospy.Publisher('/ddp/online_controller_viz', Marker)
    state_subscriber = rospy.Subscriber('/ddp/mat_controller_state', MPCState, state_callback)

    print 'Python controller initialized and spinning...'
    rospy.spin()

if __name__ == "__main__":
    main()
