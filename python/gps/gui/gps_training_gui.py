import copy
import itertools
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from gps.gui.config import common as common_config
from gps.gui.config import gps_training as gps_training_config
from gps.gui.action import Action
from gps.gui.action_axis import ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.plotter_3d import Plotter3D
from gps.gui.image_visualizer import ImageVisualizer

from gps.proto.gps_pb2 import END_EFFECTOR_POINTS

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
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(gps_training_config))
        self._hyperparams.update(hyperparams)

        self._log_filename = self._hyperparams['log_filename']

        # GPS Training Status.
        self.mode = 'run'       # Valid modes: run, wait, end, request, process.
        self.request = None     # Valid requests: stop, reset, go, fail, None.
        self.err_msg = None
        self._colors = {
            'run': 'cyan',
            'wait': 'orange',
            'end': 'red',

            'stop': 'red',
            'reset': 'yellow',
            'go': 'green',
            'fail': 'magenta',
        }
        self._first_update = True

        # Actions.
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

        # GUI Components.
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['keymap.save'] = ''  # Remove 's' keyboard shortcut for saving.

        self._fig = plt.figure(figsize=(12, 12))
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        self._gs  = gridspec.GridSpec(4, 4)

        # Action Axis.
        self._gs_action = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=self._gs[0:1,0:4])
        self._axarr_action = [plt.subplot(self._gs_action[0,i]) for i in range(4)]
        self._action_axis = ActionAxis(self._actions, self._axarr_action,
                ps3_process_rate=self._hyperparams['ps3_process_rate'],
                ps3_topic=self._hyperparams['ps3_topic'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])

        self._ax_action_output = plt.subplot(self._gs_action[1,2:4])
        self._action_output_axis = OutputAxis(self._ax_action_output, border_on=True)

        self._ax_status_output = plt.subplot(self._gs_action[1,0:2])
        self._status_output_axis = OutputAxis(self._ax_status_output)

        # Output Axis.
        self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self._gs[1:2,0:4])
        self._ax_plot = plt.subplot(self._gs_output[1])
        self._plot_axis = MeanPlotter(self._ax_plot, label='cost')

        self._ax_output = plt.subplot(self._gs_output[0])
        self._output_axis = OutputAxis(self._ax_output, max_display_size=10,
                log_filename=self._log_filename, font_family='monospace')
        for line in self._hyperparams['info'].split('\n'):
            self.append_output_text(line)

        # Visualization Axis
        num_conditions = self._hyperparams['conditions']
        rows = 2
        cols = int(np.ceil(float(num_conditions)/rows))
        self._gs_vis = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=self._gs[2:4, 2:4])
        self._axarr_vis = [plt.subplot(self._gs_vis[i/col, i%rows], projection='3d') for i in range(num_conditions)]

        # Image Axis
        self._gs_image = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 0:2])
        self._ax_image = plt.subplot(self._gs_image[0])
        self._image_axis = ImageVisualizer(self._ax_image, cropsize=(240, 240),
                rostopic=self._hyperparams['image_topic'])

        self.run_mode()
        self._fig.canvas.draw()

    # GPS Training Functions.
    def request_stop(self, event=None):
        self.request_mode('stop')

    def request_reset(self, event=None):
        self.request_mode('reset')

    def request_go(self, event=None):
        self.request_mode('go')

    def request_fail(self, event=None):
        self.request_mode('fail')

    def request_mode(self, request):
        self.mode = 'request'
        self.request = request
        self.set_action_text(self.request + ' requested')
        self.set_action_bgcolor(self._colors[self.request], alpha=0.2)

    def process_mode(self):
        self.mode = 'process'
        self.set_action_text(self.request + ' processed')
        self.set_action_bgcolor(self._colors[self.request], alpha=1.0)
        if self.err_msg:
            self.set_action_text(self.request + ' processed' + '\nERROR: ' + self.err_msg)
            self.err_msg = None
            time.sleep(1.0)
        else:
            time.sleep(0.5)
        if self.request in ('stop', 'reset', 'fail'):
            self.wait_mode()
        elif self.request == 'go':
            self.run_mode()
        self.request = None

    def wait_mode(self):
        self.mode = 'wait'
        self.set_action_text('waiting')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def run_mode(self):
        self.mode = 'run'
        self.set_action_text('running')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def end_mode(self):
        self.mode = 'end'
        self.set_action_text('ended')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def estop(self, event=None):
        self.set_action_text('estop: NOT IMPLEMENTED')

    # GUI functions.
    def set_status_text(self, text):
        self._status_output_axis.set_text(text)

    def set_output_text(self, text):
        self._output_axis.set_text(text)

    def append_output_text(self, text):
        self._output_axis.append_text(text)

    def set_action_text(self, text):
        self._action_output_axis.set_text(text)

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output_axis.set_bgcolor(color, alpha)

    def update(self, itr, algorithm, traj_sample_lists, pol_sample_lists):
        # Plot Costs
        if algorithm.M == 1:
            # Update plot with each sample's cost (summed over time).
            costs = np.sum(algorithm.prev[0].cs, axis=1)
        else:
            # Update plot with each condition's mean sample cost (summed over time).
            costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        self._plot_axis.update(costs)

        # Print Costs
        if self._first_update:
            self.set_output_text(self._hyperparams['experiment_name'])
            self.append_output_text('itr | cost')
            self._first_update = False
        self.append_output_text('%02d  | %f' % (itr, np.mean(costs)))

        # Plot Trajectory Visualizations
        for m in range(algorithm.M):
            cond_axis = self.axarr_vis[m]
            cond_axis.clear()
            cond_axis.legend()
            # Plot Trajectory Samples
            traj_samples = traj_sample_lists[m].get_samples()
            for sample in traj_samples:
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                cond_axis.plot(xs=ee_pt[:,0], ys=ee_pt[:,1], zs=ee_pt[:,2], color='green', label='Trajectory Samples')
            # Plots Policy Samples
            pol_samples = pol_sample_lists[m].get_samples()
            for sample in pol_samples:
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                cond_axis.plot(xs=ee_pt[:,0], ys=ee_pt[:,1], zs=ee_pt[:,2], color='blue', label='Policy Samples')
            # Plot Linear Gaussian Controllers (Mean/Covariance)
            # TODO
        self._fig.canvas.draw()
        self._fig.canvas.flush_events() # Fixes bug with Qt4Agg backend
