import rospy
import roslib; roslib.load_manifest('gps_agent_pkg')
from sensor_msgs.msg import Joy

import copy
import itertools

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button

from gps.gui.config import gui as gui_config
from gps.gui.config import ps3_button, inverted_ps3_button

from gps.agent.ros.agent_ros import AgentROS
from gps.gui.action_lib import ActionLib
from gps.gui.target_setup import TargetSetup
from gps.gui.training_handler import TrainingHandler
from gps.gui.real_time_plotter import RealTimePlotter
from gps.gui.image_visualizer import ImageVisualizer

# ~~~ GUI Specifications ~~~
# Target Setup (responsive to mouse, keyboard, and PS3 controller)
#     - set target number, set actuator type
#     - relax controller, mannequin mode
#     - set initial position (joint angles, end effector points), move to initial position
#     - set target  position (joint angles, end effector points), move to target  position
#     - set initial feature points, set target feature points

# Training Handler
#     - stop, stop and reset, reset, start

# Data Plotter
#     - algorithm training costs
#     - losses of feature points / end effector points
#     - joint states, feature point states, etc.
#     - save tracked data to file

# Image Visualizer
#     - real-time image and feature points visualization
#     - overlay of initial and target feature points
#     - visualize hidden states?
#     - create movie from image visualizations

class GUI:
    def __init__(self, agent, hyperparams):
        # Hyperparameters
        self._hyperparams = copy.deepcopy(gui_config)
        self._hyperparams.update(hyperparams)
        self._output_files_dir = self._hyperparams['output_files_dir']
        self._actions_log_filename = self._output_files_dir + self._hyperparams['actions_log_filename']

        # Action Lib
        ts = TargetSetup(agent, hyperparams)
        th = TrainingHandler(agent, hyperparams)
        self._action_lib = ActionLib(ts, th)
        self._actions =  self._action_lib._actions
        self._action_lib._ts._gui = self
        self._action_lib._th._gui = self

        # GUI Components
        plt.ion()
        self._fig = plt.figure(figsize=(10, 10))
        self._gs  = gridspec.GridSpec(2, 1)

        self._gs_top   = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=self._gs[0])
        self._gs_ts    = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=self._gs_top[0:3, 0:4])
        self._gs_th    = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=self._gs_top[3, 0:4])
        self._axarr_ts = [plt.subplot(self._gs_ts[i]) for i in range(3*4)]
        self._axarr_th = [plt.subplot(self._gs_th[i]) for i in range(1*4)]

        self._gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=self._gs[1])
        self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_bottom[0, 0])
        self._gs_plot   = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_bottom[1, 0])
        self._gs_vis    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_bottom[0:1, 1])
        self._ax_output = plt.subplot(self._gs_output[0])
        self._ax_plot   = plt.subplot(self._gs_plot[0])
        self._ax_vis    = plt.subplot(self._gs_vis[0])

        # Button Locations
        action_ax = {
            # Target Setup
            'ptn': self._axarr_ts[0],
            'ntn': self._axarr_ts[1],
            'pat': self._axarr_ts[2],
            'nat': self._axarr_ts[3],

            'sip': self._axarr_ts[4],
            'stp': self._axarr_ts[5],
            'sif': self._axarr_ts[6],
            'stf': self._axarr_ts[7],

            'mti': self._axarr_ts[8],
            'mtt': self._axarr_ts[9],
            'rc':  self._axarr_ts[10],
            'mm':  self._axarr_ts[11],

            # Training Handler
            'stop':  self._axarr_th[0],
            'st-re': self._axarr_th[1],
            'reset': self._axarr_th[2],
            'start': self._axarr_th[3],
        }

        # Keyboard Input
        self._keyboard_bindings = {}
        for key, keyboard_key in self._hyperparams['keyboard_bindings'].iteritems():
            self._actions[key]._kb = keyboard_key
            self._keyboard_bindings[keyboard_key] = key
        self._cid = self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Mouse Input
        self._buttons = {}
        for key, ax in action_ax.iteritems():
            action = self._actions[key]
            action._ax = ax
            self._buttons[key] = Button(action._ax, action._name + '\n' + '(' + action._kb + ')')
            self._buttons[key].on_clicked(action._func)

        # PS3 Controller Input
        self._ps3_controller_bindings = {}
        for key, ps3_controller_buttons in self._hyperparams['ps3_controller_bindings'].iteritems():
            self._actions[key]._cb = ps3_controller_buttons
            self._ps3_controller_bindings[ps3_controller_buttons] = key
        for key, value in list(self._ps3_controller_bindings.iteritems()):
            for permuted_key in itertools.permutations(key, len(key)):
                self._ps3_controller_bindings[permuted_key] = value
        self._ps3_controller_count = 0
        self._ps3_controller_message_rate = self._hyperparams['ps3_controller_message_rate']
        rospy.Subscriber(self._hyperparams['ps3_controller_topic'], Joy, self.ps3_controller_callback)

        # Output Panel
        self.set_output_text("Waiting for response from agent...")

        # Plot Panel
        self._plotter = RealTimePlotter(self._ax_plot, labels=['cost'])
        # TODO: self._plotter.update(x)

        # Visualizations Panel
        self._visualizer = ImageVisualizer(self._ax_vis, cropsize=(240,240))
        # TODO: self._visualizer.update(image)

        self._fig.canvas.draw()

    def on_key_press(self, event):
        if event.key in self._keyboard_bindings:
            self._actions[self._keyboard_bindings[event.key]]._func()
        else:
            self.set_output_text("unrecognized keyboard input: " + str(event.key))

    def ps3_controller_callback(self, joy_msg):
        self._ps3_controller_count += 1
        if self._ps3_controller_count % self._ps3_controller_message_rate != 0:
            return
        buttons_pressed = tuple([i for i in range(len(joy_msg.buttons)) if joy_msg.buttons[i]])
        if buttons_pressed in self._ps3_controller_bindings:
            self._actions[self._ps3_controller_bindings[buttons_pressed]]._func()
        else:
            if not (len(buttons_pressed) == 0 or (len(buttons_pressed) == 1 and
                    (buttons_pressed[0] == ps3_button['rear_right_1'] or buttons_pressed[0] == ps3_button['rear_right_2']))):
                self.set_output_text("unrecognized ps3 controller input: " + '\n' +
                                     str([inverted_ps3_button[b] for b in buttons_pressed]))

    def set_output_text(self, text):
        self._ax_output.clear()
        self._ax_output.set_axis_off()
        self._ax_output.text(0, 1, text, color='black', fontsize=12,
            va='top', ha='left', transform=self._ax_output.transAxes)
        self._fig.canvas.draw()
        with open(self._actions_log_filename, "a") as f:
            f.write(text + '\n\n')

    def update(self, algorithm):
        for t in range(algorithm.T):
            # update with mean cost over all samples, for each condition m, for each time step t
            self._plotter.update([np.mean(algorithm.prev[m].cs[:, t]) for m in range(algorithm.M)])

if __name__ == "__main__":
    import imp
    hyperparams = imp.load_source('hyperparams', 'experiments/default_pr2_experiment/hyperparams.py')

    rospy.init_node('gui')
    agent = AgentROS(hyperparams.config['agent'], init_node=False)
    g = GUI(agent, hyperparams.config['common'])

    plt.ioff()
    plt.show()
