import rospy
import roslib; roslib.load_manifest('gps_agent_pkg')
from sensor_msgs.msg import Joy

from datetime import datetime
import copy
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
from matplotlib.text import Text

from gps.gui.config import target_setup
from gps.gui.config import keybindings, controller_bindings
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, JOINT_ANGLES
from gps.agent.ros.agent_ros import AgentROS
#from gps_agent_pkg.msg import RelaxCommand.LEFT_ARM as ARM_LEFT
#from gps_agent_pkg.msg import RelaxCommand.RIGHT_ARM as ARM_RIGHT
from gps_agent_pkg.msg import PositionCommand
from gps.hyperparam_pr2 import defaults as agent_config

ARM_LEFT = 1
ARM_RIGHT = 2

# ~~~ GUI Specifications ~~~
# Target setup (responsive to keyboard, gui, and PS3 controller)
#   - set target number, set sensor type
#    - relax controller, mannequin mode
#   - set initial position (joint angles), move to initial position
#    - set target position (joint angles), move to target position
#    - set target end effector points, set target feature points
#
# Robot training
#   - stop, stop and reset, reset, reset and go, go
#
# Data visualizer
#    - algorithm training visualizations
#     - real-time image and feature points visualization
#    - overlay of initial and target feature points
#    - visualize hidden states?
#
# Data plotter
#    - algorithm training costs
#    - losses of feature points / end effector points
#    - joint states, feature point states, etc.
#
# Data recorder
#    - save tracked data to file
#    - create movie from image visualizations

class GUI:
    def __init__(self, agent, hyperparams):
        # General
        self._agent = agent
        self._hyperparams = copy.deepcopy(target_setup)
        self._hyperparams.update(hyperparams)
        self._filedir = self._hyperparams['file_dir']

        # Target setup
        self._target_number = 1
        self._sensor_names = {1: 'right_arm', 2: 'left_arm'}
        self._sensor_type = self._sensor_names[1]
        # self._output_file = self._filedir + "gui_output_" + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')
        self._keybindings = keybindings
        self._controller_bindings = controller_bindings
        self._keyfunctions = {
            'stn1': lambda event: self.set_target_number(1),
            'stn2': lambda event: self.set_target_number(2),
            'stn3': lambda event: self.set_target_number(3),
            'stn4': lambda event: self.set_target_number(4),
            'stn5': lambda event: self.set_target_number(5),
            'stn6': lambda event: self.set_target_number(6),
            'stn7': lambda event: self.set_target_number(7),
            'stn8': lambda event: self.set_target_number(8),
            'stn9': lambda event: self.set_target_number(9),
            'stn0': lambda event: self.set_target_number(0),
            'sst1': lambda event: self.set_sensor_type(1),
            'sst2': lambda event: self.set_sensor_type(2),
            'sip': lambda event: self.set_initial_position(event),
            'stp': lambda event: self.set_target_position(event),
            'set': lambda event: self.set_ee_target(event),
            'sft': lambda event: self.set_ft_target(event),
            'mti': lambda event: self.move_to_initial(event),
            'mtt': lambda event: self.move_to_target(event),
            'rc': lambda event: self.relax_controller(event),
            'mm': lambda event: self.mannequin_mode(event),
        }
        rospy.Subscriber("joy", Joy, self.joystick_callback)

        # GUI components
        r, c = 5, 5
        self._fig = plt.figure(figsize=(8, 8))
        self._gs  = gridspec.GridSpec(1, 2)

        self._gs_left   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self._gs[0])
        self._gs_setup  = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=self._gs_left[0])
        self._gs_train  = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=self._gs_left[1])

        self._gs_right  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self._gs[1])
        self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_right[0])
        self._gs_vis    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_right[1])

        # ~~~ SETUP PANEL ~~~
        self._actions = [('set_target_number', self.set_target_number),
                        ('set_sensor_type', self.set_sensor_type),
                        ('set_initial_position', self.set_initial_position),
                        ('move_to_initial', self.move_to_initial),
                        ('set_target_position', self.set_target_position),
                        ('move_to_target', self.move_to_target),
                        ('set_ee_target', self.set_ee_target),
                        ('relax_controller', self.relax_controller),
                        ('set_ft_target', self.set_ft_target),
                        ('mannequin_mode', self.mannequin_mode)]
        num_actions = len(self._actions)
        self._axarr = [plt.subplot(self._gs_setup[i]) for i in range(num_actions)]

        self._target_slider = DiscreteSlider(self._axarr[0], 'set_target_number', 1, 13, valinit=1, valfmt='%d')
        self._target_slider.on_changed(self.set_target_number)
        self._sensor_slider = DiscreteSlider(self._axarr[1], 'set_sensor_type', 1,  2, valinit=1, valfmt='%d')
        self._sensor_slider.on_changed(self.set_sensor_type)

        buttons_start = 2
        self._actions_button = self._actions[buttons_start:]
        num_buttons = len(self._actions_button)
        self._buttons = [Button(self._axarr[buttons_start+i], self._actions_button[i][0]) for i in range(num_buttons)]
        [self._buttons[i].on_clicked(self._actions_button[i][1]) for i in range(num_buttons)]

        # ~~~ TRAIN PANEL ~~~
        # self._actions_train = [('stop', self.stop_training),
        #                        ('stop_reset', self.stop_reset_training),
        #                        ('reset', self.reset_training),
        #                        ('start', self.start_training),]
        # self._axarr_train = [plt.subplot(self._gs_train[i]) for i in range(len(_actions_train))]
        # self._buttons_train = [Button(self._axarr_train[i], self._actions_train[i][0]) for i in range(len(_actions_train))]
        # [self._buttons_train[i].on_clicked(self._actions_train[i][1]) for i in range(len(_actions_train))]

        # ~~~ OUTPUT PANEL ~~~
        self._output_ax = plt.subplot(self._gs_output[0])
        self.set_output("target number: " +  str(self._target_number) + "\n" +
                "sensor type: " + self._sensor_type)

        # ~~~ VISUALIZATIONS PANEL ~~~

        # Keyboard Input
        self._cid = self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # PS3 Controller Input
        pass

    def on_key_press(self, event):
        if event.key in self._keybindings.keys():
            self._keyfunctions[self._keybindings[event.key]](event)
        else:
            self.set_output("unrecognized keybinding: " + event.key)

    def joystick_callback(self, joy_msg):
        buttons_pressed = tuple([i for i in range(len(joy_msg.buttons)) if joy_msg.buttons[i]])
        self._keyfunctions[self._controller_bindings[buttons_pressed]](None)

    def set_output(self, text):
        self._output_ax.clear()
        self._output_ax.set_axis_off()
        self._output_ax.text(0, 1, text, color='green', fontsize=12,
            va='top', ha='left', transform=self._output_ax.transAxes)
        self._fig.canvas.draw()
        # with open(output_file, "a") as f:
        #     f.write(text)

    # SETUP FUNCTIONS
    def set_target_number(self, val):
        discrete_val = int(val)
        self._target_number = discrete_val
        self._target_slider.update_val(discrete_val)
        self.set_output("set_target_number: " + str(self._target_number))

    def set_sensor_type(self, val):
        discrete_val = int(val)
        self._sensor_type = self._sensor_names[discrete_val]
        # self._sensor_slider.set_val(discrete_val)
        self.set_output("set_sensor_type: " + self._sensor_type)

    def relax_controller(self, event):
        self._agent.relax_arm(self._sensor_type)
        self.set_output("relax_controller: " + self._sensor_type)

    def mannequin_mode(self, event):
        # TO-DO
        self.set_output("mannequin_mode: " + "NOT YET IMPLEMENTED")

    def set_initial_position(self, event):
        x = self._agent.get_data(JOINT_ANGLES)    # TODO - this is specific to AgentROS...
        filename = self._filedir + self._sensor_type + '_initial_' + self._target_number + '.npz'
        np.savez(filename, x=x)
        self.set_output("set_initial_position: " + x)

    def move_to_initial(self, event):
        filename = self._filedir + self._sensor_type + '_initial_' + self._target_number + '.npz'
        with np.load(filename) as f:
            x = f['x']
        self._agent.reset_arm(ARM_LEFT, 1, x)
        self.set_output("move_to_initial: " + x)

    def set_target_position(self, event):
        x = self._agent.get_data(JOINT_ANGLES)    # TODO - this is specific to AgentROS...
        filename = self._filedir + self._sensor_type + '_target_' + self._target_number + '.npz'
        np.savez(filename, x=x)
        self.set_output("set_target_position: " + x)

    def move_to_target(self, event):
        filename = self._filedir + self._sensor_type + '_target_' + self._target_number + '.npz'
        with np.load(filename) as f:
            x = f['x']
        self._agent.reset_arm(ARM_LEFT, 1, x)
        self.set_output("move_to_target: " + x)

    def set_ee_target(self, event):
        x = self._agent.get_data(END_EFFECTOR_POINTS)    # TODO - this is specific to AgentROS...
        filename = self._filedir + 'ee' + '_target_' + self._target_number + '.npz'
        np.savez(filename, x=x)
        self.set_output("set_ee_target: " + x)

    def set_ft_target(self, event):
        num_samples = 50
        threshold = 0.8

        ft_points_samples = np.empty()
        ft_prsnce_samples = np.empty()
        for i in range(num_samples):
            ft_points_samples.append(self._agent.get_data(self._sensor_type, VISUAL_FEATURE_POINTS))        # currently not implemented
            ft_prsnce_samples.append(self._agent.get_data(self._sensor_type, VISUAL_FEATURE_PRESENCE))    # currently not implemented
        ft_points_mean = np.mean(ft_points)
        ft_prsnce_mean = np.mean(ft_pres)

        ft_stable = np.array(ft_prsnce_mean >= threshold, dtype=int)
        ft_points = ft_stable * ft_points_mean

        filename = self._filedir + 'ft' + '_target_' + self._target_number + '.npz'
        np.savez(filename, ft_points=ft_points, ft_stable=ft_stable)
        self.set_output("set_ft_target: " + "\n" +
                "ft_points: " + ft_points + "\n" +
                "ft_stable: " + ft_stable)

    # TRAIN FUNCTIONS
    def stop_training(self, event):
        pass

    def stop_reset_training(self, event):
        pass

    def reset_training(self, event):
        pass

    def start_training(self, event):
        pass

class DiscreteSlider(Slider):
    def __init__(self, *args, **kwargs):
        Slider.__init__(self, *args, **kwargs)

        self.label.set_transform(self.ax.transAxes)
        self.label.set_position((0.5, 0.5))
        self.label.set_ha('center')
        self.label.set_va('center')

        self.valtext.set_transform(self.ax.transAxes)
        self.valtext.set_position((0.5, 0.3))
        self.valtext.set_ha('center')
        self.valtext.set_va('center')

    def set_val(self, val):
        self.val = val
        discrete_val = round(val)
        self.valtext.set_text(self.valfmt % discrete_val)
        self.poly.xy[2] = discrete_val, 1
        self.poly.xy[3] = discrete_val, 0
        if self.drawon:
            self.ax.figure.canvas.draw()
        if self.eventson:
            for cid, func in self.observers.iteritems():
                func(discrete_val)

    def update_val(self, val):
        # self.val = val
        discrete_val = round(val)
        self.valtext.set_text(self.valfmt % discrete_val)
        self.poly.xy[2] = discrete_val, 1
        self.poly.xy[3] = discrete_val, 0
        # if self.drawon:
        #     self.ax.figure.canvas.draw()
        # if self.eventson:
        #     for cid, func in self.observers.iteritems():
        #         func(discrete_val)

if __name__ == "__main__":
    rospy.init_node('gui')
    a = AgentROS(agent_config['agent'], init_node=False)
    g = GUI(a, agent_config['gui'])
    plt.show()
