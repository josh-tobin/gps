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
    def __init__(self, agent, hyperparams):
        # Hyperparameters
        self._agent = agent
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(gps_training_config))
        self._hyperparams.update(hyperparams)

        # Output files
        self._output_files_dir = self._hyperparams['output_files_dir']
        self._log_filename = self._output_files_dir + self._hyperparams['gps_training_log_filename']

        # GPS Training
        pass

        # Actions
        actions_arr = [
            Action('stop',  'stop',     self.stop,  axis_pos=0),
            Action('reset', 'reset',    self.reset, axis_pos=1),
            Action('start', 'start',    self.start, axis_pos=2),
            Action('estop', 'estop',    self.estop, axis_pos=3),
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
        self._fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        self._gs  = gridspec.GridSpec(4, 4)

        # Action Axis
        self._gs_action = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=self._gs[0:1, 0:4])
        self._axarr_action = [plt.subplot(self._gs_action[i]) for i in range(1*4)]
        self._action_axis = ActionAxis(self._actions, self._axarr_action,
                ps3_process_rate=self._hyperparams['ps3_process_rate'], ps3_topic=self._hyperparams['ps3_topic'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])

        # Output Axis
        self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[1:2, 0:4])
        self._ax_output = plt.subplot(self._gs_output[0])
        self._output_axis = OutputAxis(self._ax_output, max_display_size=5, log_filename=self._log_filename)

        # Plot Axis
        self._gs_plot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 0:2])
        self._ax_plot = plt.subplot(self._gs_plot[0])
        self._plot_axis = MeanPlotter(self._ax_plot, label=['cost'])

        # Image Axis
        self._gs_image = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 2:4])
        self._ax_image = plt.subplot(self._gs_image[0])
        self._visualizer = ImageVisualizer(self._ax_image, cropsize=(240,240), rostopic=self._hyperparams['image_topic'])

        self._fig.canvas.draw()

    # GPS Training Functions
    def stop(self, event=None):
        self.set_text('stop')
        self.set_bgcolor('red')
        pass

    def reset(self, event=None):
        self.set_text('reset')
        self.set_bgcolor('yellow')
        pass

    def start(self, event=None):
        self.set_text('start')
        self.set_bgcolor('green')
        pass

    def estop(self, event=None):
        self.set_text('estop')
        for i in range(10):
            self.set_bgcolor('red')
            time.sleep(0.3)
            self.set_bgcolor('white')
            time.sleep(0.3)
        self.set_bgcolor('red')
        pass

    # GUI functions
    def set_text(self, text):
        self._output_axis.set_text(text)

    def set_bgcolor(self, color):
        self._output_axis.set_bgcolor(color)

    def append_text(self, text):
        self._output_axis.append_text(text)

    def update(self, algorithm):
        if algorithm.M == 1:
            # update with each sample's cost (summed over time)
            self._plot_axis.update(np.sum(algorithm.prev[0].cs, axis=1))
        else:
            # update with each condition's mean sample cost (summed over time)
            self._plot_axis.update([np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)])
