import sys
import imp

sys.path.append('/home/jt/gps/python')
import mjcpy

from gps.gui.target_setup_gui import load_pose_from_npz


EE_POINTS = np.array([[0.02, -0.025, 0.05], [0.02, -0.025, -0.05], [0.02, 0.05, 0.0]])

world = mjcpy.MJCWorld('mjc_models/PR2/pr2_1arm.xml')
model = world.get_model()
world.kinematics()
print world.get_data()['site_xpos']

ja_x0, ee_pos_x0, ee_rot_x0 = load_pose_from_npz(
        '/home/jt/gps/experiments/pr2_example/target.npz', 'trial_arm', '0', 'initial'
)

print 'j_x0 is:'
print ja_x0
data = {'qpos': ja_x0, 'qvel': np.zeros_like(ja_x0)}
world.set_data(data)
world.kinematics()
print 'MJC Kinematics:'
print world.get_data()['site_xpos']
print 'ROS Kinematics:'
print ee_pos_x0
