"""
~~~ GUI Specifications ~~~
Action Axis
    - stop, reset, start, emergency stop

Data Plotter
    - algorithm training costs
    - losses of feature points / end effector points
    - joint states, feature point states, etc.
    - save tracked data to file

Image Visualizer
    - real-time image and feature points visualization
    - overlay of initial and target feature points
    - visualize hidden states?
    - create movie from image visualizations
"""
import copy
import time

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from gps.gui.config import common as common_config
from gps.gui.config import gps_training as gps_training_config
from gps.gui.action_axis import Action, ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.three_d_plotter import ThreeDPlotter
from gps.gui.image_visualizer import ImageVisualizer

from gps.gui.target_setup_gui import load_data_from_npz
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS

class GPSTrainingGUI(object):
    """ GPS Training GUI class. """
    def __init__(self, hyperparams):
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(gps_training_config))
        self._hyperparams.update(hyperparams)

        self._log_filename = self._hyperparams['log_filename']
        self._target_filename = self._hyperparams['target_filename']

        # GPS Training Status.
        self.mode = 'run'  # Modes: run, wait, end, request, process.
        self.request = None  # Requests: stop, reset, go, fail, None.
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
            Action('stop', 'stop', self.request_stop, axis_pos=0),
            Action('reset', 'reset', self.request_reset, axis_pos=1),
            Action('go', 'go', self.request_go, axis_pos=2),
            Action('fail', 'fail', self.request_fail, axis_pos=3),
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
        # Remove 's' keyboard shortcut for saving.
        plt.rcParams['keymap.save'] = ''

        self._fig = plt.figure(figsize=(12, 12))
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        
        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(16, 8)
        self._gs_action_axis        = self._gs[0:2,  0:8]
        self._gs_action_output      = self._gs[2:3,  0:4]
        self._gs_status_output      = self._gs[3:4,  0:4]
        self._gs_cost_plotter       = self._gs[2:4,  4:8]
        self._gs_algthm_output      = self._gs[4:8,  0:8]
        self._gs_traj_visualizer    = self._gs[8:16, 0:4]
        self._gs_image_visualizer   = self._gs[8:16, 4:8]

        # Create GUI components.
        self._action_axis = ActionAxis(self._fig, self._gs_action_axis, 1, 4, self._actions,
                ps3_process_rate=self._hyperparams['ps3_process_rate'],
                ps3_topic=self._hyperparams['ps3_topic'],
                ps3_button=self._hyperparams['ps3_button'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])
        self._action_output = OutputAxis(self._fig, self._gs_action_output, border_on=True)
        self._status_output = OutputAxis(self._fig, self._gs_status_output, border_on=False)
        self._algthm_output = OutputAxis(self._fig, self._gs_algthm_output, max_display_size=10,
                log_filename=self._log_filename, font_family='monospace')
        self._cost_plotter = MeanPlotter(self._fig, self._gs_cost_plotter, label='cost')
        self._traj_visualizer = ThreeDPlotter(self._fig, self._gs_traj_visualizer, num_plots=self._hyperparams['conditions'])
        self._image_visualizer = ImageVisualizer(self._hyperparams, self._fig, self._gs_image_visualizer, 
                cropsize=(240, 240), rostopic=self._hyperparams['image_topic'], show_overlay_buttons=True)

        # Setup GUI components.
        for line in self._hyperparams['info'].split('\n'):
            self.append_output_text(line)

        self.run_mode()
        self._fig.canvas.draw()

    # GPS Training Functions.
    #TODO: Docstrings here.
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
            self.set_action_text(self.request + ' processed' + '\nERROR: ' +
                                 self.err_msg)
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
        self._status_output.set_text(text)

    def set_output_text(self, text):
        self._algthm_output.set_text(text)

    def append_output_text(self, text):
        self._algthm_output.append_text(text)

    def set_action_text(self, text):
        self._action_output.set_text(text)

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)

    def set_image_overlays(self, condition):
        initial_image = load_data_from_npz(self._target_filename, self._hyperparams['image_actuator'], str(condition), 
                'initial', 'image', default=np.zeros((1,1,3)))
        target_image  = load_data_from_npz(self._target_filename, self._hyperparams['image_actuator'], str(condition), 
                'target',  'image', default=np.zeros((1,1,3)))
        self._image_visualizer.set_initial_image(initial_image, alpha=0.3)
        self._image_visualizer.set_target_image(target_image, alpha=0.3)
        
    def update(self, itr, algorithm, traj_sample_lists, pol_sample_lists):
        # Plot Costs
        if algorithm.M == 1:
            # Update plot with each sample's cost (summed over time).
            costs = np.sum(algorithm.prev[0].cs, axis=1)
        else:
            # Update plot with each condition's mean sample cost (summed over time).
            costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        self._cost_plotter.update(costs)

        # Print Iteration Data
        if self._first_update:
            self.set_output_text(self._hyperparams['experiment_name'])
            itr_data_fields = 'itr | cost '
            for m in range(len(costs)):
                itr_data_fields += ' | cost%d' % m
            for m in range(algorithm.M):
                itr_data_fields += ' | step%d' % m
            if algorithm.prev[0].pol_info is not None:
                for m in range(algorithm.M):
                    itr_data_fields += ' | kl_div%d ' % m
            self.append_output_text(itr_data_fields)
            self._first_update = False
        itr_data = ' %02d | %3.1f' % (itr, np.mean(costs))
        for cost in costs:
            itr_data += ' | %3.1f' % cost
        for m in range(algorithm.M):
            itr_data += ' | %3.1f' % algorithm.prev[m].step_mult
        if algorithm.prev[m].pol_info is not None:
            for m in range(algorithm.M):
                itr_data += ' | %3.1f' % algorithm.prev[m].pol_info.prev_kl 
        self.append_output_text(itr_data)

        # Plot 3D Visualizations
        for m in range(algorithm.M):
            # Clear previous plots
            self._traj_visualizer.clear(m)
            # Plot Trajectory Samples
            traj_samples = traj_sample_lists[m].get_samples()
            for sample in traj_samples:
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                self.plot_3d_points(m, ee_pt, color='green', label='Trajectory Samples')
            
            # Plot Policy Samples
            if pol_sample_lists is not None:
                pol_samples = pol_sample_lists[m].get_samples()
                for sample in pol_samples:
                    ee_pt = sample.get(END_EFFECTOR_POINTS)
                    self.plot_3d_points(m, ee_pt, color='blue', label='Policy Samples')
            
            # Plot Linear Gaussian Controllers (Mean/Covariance)
            mu, sigma = algorithm.traj_opt.forward(algorithm.prev[m].traj_distr, algorithm.prev[m].traj_info)
            # TODO: gracefully extract mu and sigma for end effector points
            mu_eept, sigma_eept = mu[:, 21:27], sigma[:, 21:27] # THIS IS HARDCODED FOR MJC_EXAMPLES
            self.plot_3d_points(m, mu_eept, color='red', label='LQG Controllers')
            # look at update draw for plotting
            
            # Draw new plots
            self._traj_visualizer.draw()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events() # Fixes bug with Qt4Agg backend

    def plot_3d_points(self, m, points, color, label):
        """
        Plots a (T x (3n)) array of points in 3D, where n is an integer.
        """
        n = points.shape[1]/3
        for i in range(n):
            self._traj_visualizer.plot(
                i=m,
                xs=points[:,3*i+0],
                ys=points[:,3*i+1],
                zs=points[:,3*i+2],
                color=color,
                label=label
            )

