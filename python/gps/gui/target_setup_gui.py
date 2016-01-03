import copy
import itertools
import os.path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.config import common as common_config
from gps.gui.config import target_setup as target_setup_config
from gps.gui.action import Action
from gps.gui.action_axis import ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.image_visualizer import ImageVisualizer

from gps.proto.gps_pb2 import END_EFFECTOR_POSITIONS, END_EFFECTOR_ROTATIONS, JOINT_ANGLES, TRIAL_ARM, AUXILIARY_ARM, TASK_SPACE, JOINT_SPACE

# ~~~ GUI Specifications ~~~
# Action Axis
#     - previous target number, next target number
#     - previous actuator type, next actuator type
#     - set initial position, set target position
#     - set initial features, set target features
#     - move to initial position, move to target position
#     - relax controller, mannequin mode

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

class TargetSetupGUI:
    def __init__(self, agent, hyperparams):
        # Hyperparameters
        self._agent = agent
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(target_setup_config))
        self._hyperparams.update(hyperparams)

        # Output files
        self._output_files_dir = self._hyperparams['output_files_dir']
        self._target_files_dir = self._hyperparams['target_files_dir']
        self._log_filename = self._output_files_dir + self._hyperparams['target_setup_log_filename']

        # Target Setup
        self._num_targets = self._hyperparams['num_targets']
        self._actuator_types = self._hyperparams['actuator_types']
        self._actuator_names = self._hyperparams['actuator_names']
        self._num_actuators = len(self._actuator_types)

        self._target_number = 0
        self._actuator_number = 0
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]

        # Actions
        actions_arr = [
            Action('ptn',   'prev_target_number',       self.prev_target_number,    axis_pos=0),
            Action('ntn',   'next_target_number',       self.next_target_number,    axis_pos=1),
            Action('pat',   'prev_actuator_type',       self.prev_actuator_type,    axis_pos=2),
            Action('nat',   'next_actuator_type',       self.next_actuator_type,    axis_pos=3),

            Action('sip',   'set_initial_position',     self.set_initial_position,  axis_pos=4),
            Action('stp',   'set_target_position',      self.set_target_position,   axis_pos=5),
            Action('sif',   'set_initial_features',     self.set_initial_features,  axis_pos=6),
            Action('stf',   'set_target_features',      self.set_target_features,   axis_pos=7),

            Action('mti',   'move_to_initial',          self.move_to_initial,       axis_pos=8),
            Action('mtt',   'move_to_target',           self.move_to_target,        axis_pos=9),
            Action('rc',    'relax_controller',         self.relax_controller,      axis_pos=10),
            Action('mm',    'mannequin_mode',           self.mannequin_mode,        axis_pos=11),
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

        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        self._fig.canvas.toolbar.pack_forget()
        plt.rcParams['keymap.save'] = ''    # remove 's' keyboard shortcut for saving

        # Action Axis
        self._gs_action = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=self._gs[0:2, 0:4])
        self._axarr_action = [plt.subplot(self._gs_action[i]) for i in range(3*4)]
        self._action_axis = ActionAxis(self._actions, self._axarr_action, 
                ps3_process_rate=self._hyperparams['ps3_process_rate'], ps3_topic=self._hyperparams['ps3_topic'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])

        # Output Axis
        self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 0:2])
        self._ax_output = plt.subplot(self._gs_output[0])
        self._output_axis = OutputAxis(self._ax_output, max_display_size=5, log_filename=self._log_filename)

        # Image Axis
        self._gs_image = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 2:4])
        self._ax_image = plt.subplot(self._gs_image[0])
        self._visualizer = ImageVisualizer(self._ax_image, cropsize=(240,240), rostopic=self._hyperparams['image_topic'])

        self._fig.canvas.draw()

    # Target Setup Functions
    def prev_target_number(self, event=None):
        self._target_number = (self._target_number - 1) % self._num_targets
        self.set_text(
                'prev_target_number:' + '\n' +
                'target number = ' + str(self._target_number))

    def next_target_number(self, event=None):
        self._target_number = (self._target_number + 1) % self._num_targets
        self.set_text(
                'next_target_number:' + '\n' +
                'target number = ' + str(self._target_number))

    def prev_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number - 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.set_text(
                'prev_actuator_type:' + '\n' +
                'actuator type = ' + str(self._actuator_type) + '\n' +
                'actuator name = ' + str(self._actuator_name))

    def next_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number + 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.set_text(
                'next_actuator_type:' + '\n' +
                'actuator type = ' + str(self._actuator_type) + '\n' +
                'actuator name = ' + str(self._actuator_name))

    def set_initial_position(self, event=None):
        sample = self._agent.get_data(arm=self._actuator_type)
        filename = self._target_files_dir + self._actuator_name + '_initial.npz'

        ja_key = 'ja' + str(self._target_number)
        ja_val = sample.get(JOINT_ANGLES)
        add_to_npz(filename, ja_key, ja_val)

        ee_pos_key = 'ee_pos' + str(self._target_number)
        ee_pos_val = sample.get(END_EFFECTOR_POSITIONS)
        add_to_npz(filename, ee_pos_key, ee_pos_val)

        ee_rot_key = 'ee_rot' + str(self._target_number)
        ee_rot_val = sample.get(END_EFFECTOR_ROTATIONS)
        add_to_npz(filename, ee_rot_key, ee_rot_val)

        self.set_text(
                'set_initial_position:' + '\n' +
                'filename = ' + filename + '\n' +
                ja_key + ' = ' + str(ja_value.T) + '\n' +
                ee_pos_key + ' = ' + str(ee_pos_val) + '\n' +
                ee_rot_key + ' = ' + str(ee_rot_val))

    def set_target_position(self, event=None):
        """
        Grabs the current end effector points and joint angles of the trial
        arm and saves to target file.
        """
        sample = self._agent.get_data(arm=self._actuator_type)
        filename = self._target_files_dir + self._actuator_name + '_target.npz'

        ja_key = 'ja' + str(self._target_number)
        ja_val = sample.get(JOINT_ANGLES)
        add_to_npz(filename, ja_key, ja_val)

        ee_pos_key = 'ee_pos' + str(self._target_number)
        ee_pos_val = sample.get(END_EFFECTOR_POSITIONS)
        add_to_npz(filename, ee_pos_key, ee_pos_val)

        ee_rot_key = 'ee_rot' + str(self._target_number)
        ee_rot_val = sample.get(END_EFFECTOR_ROTATIONS)
        add_to_npz(filename, ee_rot_key, ee_rot_val)

        self.set_text(
                'set_target_position:' + '\n' +
                'filename = ' + filename + '\n' +
                ja_key + ' = ' + str(ja_value.T) + '\n' +
                ee_pos_key + ' = ' + str(ee_pos_val) + '\n' +
                ee_rot_key + ' = ' + str(ee_rot_val))

    def set_initial_features(self, event=None):
        pass

    def set_target_features(self, event=None):
        num_samples = 50
        threshold = 0.8

        ft_points_samples = np.empty()
        ft_prsnce_samples = np.empty()
        for i in range(num_samples):
            ft_points_samples.append(self._agent.get_data(self._actuator_type, VISUAL_FEATURE_POINTS))        # currently not implemented
            ft_prsnce_samples.append(self._agent.get_data(self._actuator_type, VISUAL_FEATURE_PRESENCE))    # currently not implemented
        ft_points_mean = np.mean(ft_points)
        ft_prsnce_mean = np.mean(ft_pres)

        ft_stable = np.array(ft_prsnce_mean >= threshold, dtype=int)
        ft_points = ft_stable * ft_points_mean

        filename = self._target_files_dir + self._actuator_name + '_target.npz'

        fp_key = 'fp' + str(self._target_number)
        fp_value = ft_points
        add_to_npz(filename, fp_key, fp_value)

        fs_key = 'fs' + str(self._target_number)
        fs_value = ft_stable
        add_to_npz(filename, fs_key, fs_value)

        self.set_text(
                'set_target_features' + '\n' +
                'filename = ' + filename + '\n' +
                fp_key + ' = ' + str(fp_value.T) + '\n' +
                fs_key + ' = ' + str(fs_value.T))

    def move_to_initial(self, event=None):
        filename = self._target_files_dir + self._actuator_name + '_initial.npz'
        ja_key = 'ja' + str(self._target_number)
        with np.load(filename) as f:
            ja_value = f[ja_key]
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=ja_value)
        self.set_text(
                'move_to_initial:' + '\n' +
                ja_key + ' = ' + str(ja_value.T))

    def move_to_target(self, event=None):
        filename = self._target_files_dir + self._actuator_name + '_target.npz'
        ja_key = 'ja' + str(self._target_number)
        with np.load(filename) as f:
            ja_value = f[ja_key]
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=ja_value)
        self.set_text(
                'move_to_target:' + '\n' +
                ja_key + ' = ' + str(ja_value.T))

    def relax_controller(self, event=None):
        self._agent.relax_arm(arm=self._actuator_type)
        self.set_text(
                'relax_controller:' + '\n' +
                'actuator type = ' + str(self._actuator_type) + '\n' +
                'actuator name = ' + str(self._actuator_name))

    def mannequin_mode(self, event=None):
        # TO-DO
        self.set_text(
                'mannequin_mode:' + '\n' +
                'NOT YET IMPLEMENTED')

    # GUI functions
    def set_text(self, text):
        self._output_axis.set_text(text)

    def append_text(self, text):
        self._output_axis.append_text(text)

    def set_bgcolor(self, color):
        self._output_axis.set_bgcolor(color)

def add_to_npz(filename, key, value):
    """
    Helper function for adding a new (key,value) pair to a npz dictionary.

    Note: key must be a string and value must be a numpy array.
    """
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename,**tmp)

def load_from_npz(filename, key, default_dim=14):
    """
    Helper function for loading a target setup value from a npz dictionary.
    """
    try:
        with np.load(filename) as f:
            return f[key]
    except IOError as e:
        print('File not found: ' + filename + '\n' +
              'Using default value instead: ' + key + ' = np.zeros('+str(default_dim)+').')
    return np.zeros(default_dim)

if __name__ == "__main__":
    import rospy
    from gps.agent.ros.agent_ros import AgentROS

    import imp
    hyperparams = imp.load_source('hyperparams', 'experiments/default_pr2_experiment/hyperparams.py')

    rospy.init_node('target_setup_gui')
    agent = AgentROS(hyperparams.config['agent'], init_node=False)
    target_setup_gui = TargetSetupGUI(agent, hyperparams.config['common'])

    plt.ioff()
    plt.show()
