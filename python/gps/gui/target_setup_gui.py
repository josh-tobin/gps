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
    def __init__(self, hyperparams, agent):
        # Hyperparameters
        self._hyperparams = copy.deepcopy(common_config)
        self._hyperparams.update(copy.deepcopy(target_setup_config))
        self._hyperparams.update(hyperparams)
        self._agent = agent

        self._log_filename = self._hyperparams['log_filename']
        self._target_filename = self._hyperparams['target_filename']

        self._num_targets = self._hyperparams['num_targets']
        self._actuator_types = self._hyperparams['actuator_types']
        self._actuator_names = self._hyperparams['actuator_names']
        self._num_actuators = len(self._actuator_types)

        # Target Setup Status
        self._target_number = 0
        self._actuator_number = 0
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self._initial_position = ('unknown', 'unknown', 'unknown')  # (ja, eepos, eerot)
        self._target_position  = ('unknown', 'unknown', 'unknown')  # (ja, eepos, eerot)
        self.reload_positions()

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
        plt.rcParams['toolbar'] = 'None'
        plt.rcParams['keymap.save'] = ''    # remove 's' keyboard shortcut for saving

        self._fig = plt.figure(figsize=(12, 12))
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        self._gs  = gridspec.GridSpec(4, 4)

        # Action Axis
        self._gs_action = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=self._gs[0:2, 0:4])
        self._axarr_action = [plt.subplot(self._gs_action[i]) for i in range(3*4)]
        self._action_axis = ActionAxis(self._actions, self._axarr_action, 
                ps3_process_rate=self._hyperparams['ps3_process_rate'], ps3_topic=self._hyperparams['ps3_topic'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])

        # Output Axis
        self._gs_output = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=self._gs[2:4, 0:2])

        self._ax_action_output = plt.subplot(self._gs_output[3])
        self._action_output_axis = OutputAxis(self._ax_action_output)

        self._ax_status_output = plt.subplot(self._gs_output[0:3])
        self._status_output_axis = OutputAxis(self._ax_status_output, log_filename=self._log_filename)
        self.update_status_text()

        # Image Axis
        self._gs_image = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[2:4, 2:4])
        self._ax_image = plt.subplot(self._gs_image[0])
        self._visualizer = ImageVisualizer(self._ax_image, cropsize=(240,240), rostopic=self._hyperparams['image_topic'])

        self._fig.canvas.draw()

    # Target Setup Functions
    def prev_target_number(self, event=None):
        self._target_number = (self._target_number - 1) % self._num_targets      
        self.reload_positions()
        self.update_status_text()
        self.set_action_text()

    def next_target_number(self, event=None):
        self._target_number = (self._target_number + 1) % self._num_targets        
        self.reload_positions()
        self.update_status_text()
        self.set_action_text()

    def prev_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number - 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.reload_positions()
        self.update_status_text()
        self.set_action_text()

    def next_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number + 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.reload_positions()
        self.update_status_text()
        self.set_action_text()

    def set_initial_position(self, event=None):
        sample = self._agent.get_data(arm=self._actuator_type)
        ja = sample.get(JOINT_ANGLES)
        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)
        self._initial_position = (ja, ee_pos, ee_rot)
        save_pose_to_npz(self._target_filename, self._actuator_name, str(self._target_number), 'initial', self._initial_position)
        self.update_status_text()
        self.set_action_text('set_initial_position: success')

    def set_target_position(self, event=None):
        sample = self._agent.get_data(arm=self._actuator_type)
        ja = sample.get(JOINT_ANGLES)
        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)
        self._target_position = (ja, ee_pos, ee_rot)
        save_pose_to_npz(self._target_filename, self._actuator_name, str(self._target_number), 'target', self._target_position)
        
        self.update_status_text()
        self.set_action_text('set_target_position: success')

    def set_initial_features(self, event=None):
        self.set_action_text('set_initial_features: NOT IMPLEMENTED')

    def set_target_features(self, event=None):
        self.set_action_text('set_target_features: NOT IMPLEMENTED')
        # num_samples = 50
        # threshold = 0.8

        # ft_points_samples = np.empty()
        # ft_prsnce_samples = np.empty()
        # for i in range(num_samples):
        #     ft_points_samples.append(self._agent.get_data(self._actuator_type, VISUAL_FEATURE_POINTS))        # currently not implemented
        #     ft_prsnce_samples.append(self._agent.get_data(self._actuator_type, VISUAL_FEATURE_PRESENCE))    # currently not implemented
        # ft_points_mean = np.mean(ft_points)
        # ft_prsnce_mean = np.mean(ft_pres)

        # ft_stable = np.array(ft_prsnce_mean >= threshold, dtype=int)
        # ft_points = ft_stable * ft_points_mean

        # filename = self._target_files_dir + self._actuator_name + '_target.npz'

        # fp_key = 'fp' + str(self._target_number)
        # fp_value = ft_points
        # add_to_npz(filename, fp_key, fp_value)

        # fs_key = 'fs' + str(self._target_number)
        # fs_value = ft_stable
        # add_to_npz(filename, fs_key, fs_value)

        # self.set_text(
        #         'set_target_features' + '\n' +
        #         'filename = ' + filename + '\n' +
        #         fp_key + ' = ' + str(fp_value.T) + '\n' +
        #         fs_key + ' = ' + str(fs_value.T))

    def move_to_initial(self, event=None):
        ja = self._initial_position[0]
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=ja)
        self.set_action_text('move_to_initial: %s' % (str(ja.T)))

    def move_to_target(self, event=None):
        ja = self._target_position[0]
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=ja)
        self.set_action_text('move_to_target: %s' % (str(ja.T)))

    def relax_controller(self, event=None):
        self._agent.relax_arm(arm=self._actuator_type)
        self.set_action_text('relax_controller: %s' % (self._actuator_name))

    def mannequin_mode(self, event=None):
        self.set_action_text('mannequin_mode: NOT IMPLEMENTED')

    # GUI functions
    def update_status_text(self):
        text = ('target number = %s\n' % (str(self._target_number)) +
                'actuator name = %s\n' % (str(self._actuator_name)) +
                'initial position\n    ja = %s\n    ee_pos = %s\n    ee_rot = %s\n' % self._initial_position +
                'target position \n    ja = %s\n    ee_pos = %s\n    ee_rot = %s\n' % self._target_position)
        self._status_output_axis.set_text(text)

    def set_action_text(self, text=''):
        self._action_output_axis.set_text(text)

    def reload_positions(self):
        self._initial_position = load_pose_from_npz(self._target_filename, self._actuator_name, str(self._target_number), 'initial')
        self._target_position  = load_pose_from_npz(self._target_filename, self._actuator_name, str(self._target_number), 'target')

def save_pose_to_npz(filename, actuator_name, target_number, data_time, values):
    ja, ee_pos, ee_rot = values
    save_data_to_npz(filename, actuator_name, target_number, data_time, 'ja',     ja)
    save_data_to_npz(filename, actuator_name, target_number, data_time, 'ee_pos', ee_pos)
    save_data_to_npz(filename, actuator_name, target_number, data_time, 'ee_rot', ee_rot)

def save_data_to_npz(filename, actuator_name, target_number, data_time, data_name, value):
    """
    Save data to the specified file with key (actuator_name, target_number, data_time, data_name)
    """
    key = '_'.join((actuator_name, target_number, data_time, data_name))
    save_to_npz(filename, key, value)

def save_to_npz(filename, key, value):
    """
    Save a (key,value) pair to a npz dictionary.

    filename    - the file containing the npz dictionary
    key         - the key (string)
    value       - the value (numpy array)
    """
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename,**tmp)

def load_pose_from_npz(filename, actuator_name, target_number, data_time):
    ja     = load_data_from_npz(filename, actuator_name, target_number, data_time, 'ja',     default=np.zeros(7))
    ee_pos = load_data_from_npz(filename, actuator_name, target_number, data_time, 'ee_pos', default=np.zeros(3))
    ee_rot = load_data_from_npz(filename, actuator_name, target_number, data_time, 'ee_rot', default=np.eye(3))
    return (ja, ee_pos, ee_rot)

def load_data_from_npz(filename, actuator_name, target_number, data_time, data_name, default=None):
    """
    Load data from the specified file with key (actuator_name, target_number, data_time, data_name)
    """
    key = '_'.join((actuator_name, target_number, data_time, data_name))
    return load_from_npz(filename, key, default)

def load_from_npz(filename, key, default=None):
    """
    Load a (key,value) pair from a npz dictionary.

    filename    - the file containing the npz dictionary
    key         - the key (string)
    default     - the default value to return, if key or file not found
    """
    try:
        with np.load(filename) as f:
            return f[key]
    except (IOError, KeyError) as e:
        print 'error loading %s from %s' % (key, filename), e
        pass
    return default

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
