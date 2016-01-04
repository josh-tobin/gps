import copy
import itertools
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.config import common as common_config
from gps.gui.config import gps_training as gps_training_config
from gps.gui.action import Action
from gps.gui.action_axis import ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.image_visualizer import ImageVisualizer

# ~~~ GUI Specifications ~~~
# Action Axis
#     - stop, reset, start, emergency stop

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

class GPSTrainingGUI:
    def __init__(self, hyperparams):
        # Hyperparameters
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(gps_training_config))
        self._hyperparams.update(hyperparams)

        self._log_filename = self._hyperparams['log_filename']
        
        # GPS Training Status
        self.request = None
        self.request_color = None

        # Actions
        actions_arr = [
            Action('stop',  'stop',  self.request_stop,  axis_pos=0),
            Action('reset', 'reset', self.request_reset, axis_pos=1),
            Action('go',    'go',    self.request_go,    axis_pos=2),
            Action('fail',  'fail',  self.request_fail,  axis_pos=3),
        ]
        self._actions = {action._key: action for action in actions_arr}
        for key, action in self._actions.iteritems():
            if key in self._hyperparams['keyboard_bindings']:
                action._kb = self._hyperparams['keyboard_bindings'][key]
            if key in self._hyperparams['ps3_bindings']:
                action._pb = self._hyperparams['ps3_bindings'][key]

        # GUI Components
        plt.ion()
        self._fig = plt.figure(figsize=(10, 10))
        self._gs  = gridspec.GridSpec(4, 4)

        self._fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
        self._fig.canvas.toolbar.pack_forget()
        plt.rcParams['keymap.save'] = ''    # remove 's' keyboard shortcut for saving

        # Action Axis
        self._gs_action = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=self._gs[0:1, 0:4])
        self._axarr_action = [plt.subplot(self._gs_action[0, i]) for i in range(1*4)]
        self._action_axis = ActionAxis(self._actions, self._axarr_action,
                ps3_process_rate=self._hyperparams['ps3_process_rate'], ps3_topic=self._hyperparams['ps3_topic'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])

        self._ax_action_output = plt.subplot(self._gs_action[1, 2:4])
        self._action_output_axis = OutputAxis(self._ax_action_output, log_filename=self._log_filename, border_on=True)

        self._ax_status_output = plt.subplot(self._gs_action[1, 0:2])
        self._status_output_axis = OutputAxis(self._ax_status_output, log_filename=self._log_filename)

        # Output Axis
        self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self._gs[1:2, 0:4])
        self._ax_plot = plt.subplot(self._gs_output[0])
        self._plot_axis = MeanPlotter(self._ax_plot, label='cost')

        self._ax_output = plt.subplot(self._gs_output[1])
        self._output_axis = OutputAxis(self._ax_output, max_display_size=10, log_filename=self._log_filename)

        # Visualization Axis
        self._gs_vis = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 0:2])
        self._ax_vis = plt.subplot(self._gs_vis[0])
        self._vis_axis = ImageVisualizer(self._ax_vis, cropsize=(240,240), rostopic=self._hyperparams['image_topic'])

        # Image Axis
        self._gs_image = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 2:4])
        self._ax_image = plt.subplot(self._gs_image[0])
        self._image_axis = ImageVisualizer(self._ax_image, cropsize=(240,240), rostopic=self._hyperparams['image_topic'])

        self._fig.canvas.draw()

    # GPS Training Functions
    def request_stop(self, event=None):
        self.request_action('stop', 'red')

    def request_reset(self, event=None):
        self.request_action('reset', 'yellow')

    def request_go(self, event=None):
        self.request_action('go', 'green')

    def request_fail(self, event=None):
        self.request_action('fail', 'orange')

    def request_action(self, request, request_color):
        self.request = request
        self.request_color = request_color
        self.set_action_text(request + ' requested')
        self.set_action_bgcolor(request_color, alpha=0.2)

    def receive(self):
        self.receive_action(self.request, self.request_color)

    def receive_action(self, request, request_color):
        self.set_action_text(request + 'received')
        self.set_action_bgcolor(request_color, alpha=1.0)
        self.request = None
        self.request_color = None
        time.sleep(0.25)

    def waiting_mode(self):
        self.request = 'wait'
        self.request_color = 'orange'
        self.set_action_text('waiting')
        self.set_action_bgcolor(self.request_color, alpha=1.0)

    def running_mode(self):
        self.request = None
        self.request_color = None
        self.set_action_text('running')
        self.set_action_bgcolor('cyan', alpha=1.0)

    def estop(self, event=None):
        self.set_action_text('estop: NOT IMPLEMENTED')
        # self.set_action_text('estop')
        # for i in range(10):
        #     self.set_action_bgcolor('red')
        #     time.sleep(0.3)
        #     self.set_action_bgcolor('white')
        #     time.sleep(0.3)
        # self.set_action_bgcolor('red')

    # GUI functions
    def set_status_text(self, text):
        self._status_output_axis.set_text(text)

    def set_action_text(self, text):
        self._action_output_axis.set_text(text)

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output_axis.set_bgcolor(color, alpha)

    def update(self, algorithm):
        if algorithm.M == 1:
            # update with each sample's cost (summed over time)
            self._plot_axis.update(np.sum(algorithm.prev[0].cs, axis=1))
        else:
            # update with each condition's mean sample cost (summed over time)
            self._plot_axis.update([np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)])
