import copy
import itertools
import os.path

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.config import common as common_config
from gps.gui.config import target_setup as target_setup_config
from gps.gui.action_axis import Action, ActionAxis
from gps.gui.output_axis import OutputAxis
from gps.gui.image_visualizer import ImageVisualizer

from gps.proto.gps_pb2 import END_EFFECTOR_POSITIONS, END_EFFECTOR_ROTATIONS, JOINT_ANGLES, TRIAL_ARM, AUXILIARY_ARM, TASK_SPACE, JOINT_SPACE
try:
    from gps.agent.ros.ros_utils import TimeoutException
except ImportError as e:
    print('Cannot import TimeoutException because ROS is not installed.')

# ~~~ GUI Specifications ~~~
# Action Axis
#     - previous target number, next target number
#     - previous actuator type, next actuator type
#     - set initial position, set target position
#     - set initial image, set target image
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

        # Target Setup Status.
        self._target_number = 0
        self._actuator_number = 0
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self._initial_position = ('unknown', 'unknown', 'unknown')
        self._target_position  = ('unknown', 'unknown', 'unknown')
        self._initial_image = None
        self._target_image  = None
        
        # Actions.
        actions_arr = [
            Action('ptn', 'prev_target_number',   self.prev_target_number,   axis_pos=0),
            Action('ntn', 'next_target_number',   self.next_target_number,   axis_pos=1),
            Action('pat', 'prev_actuator_type',   self.prev_actuator_type,   axis_pos=2),
            Action('nat', 'next_actuator_type',   self.next_actuator_type,   axis_pos=3),

            Action('sip', 'set_initial_position', self.set_initial_position, axis_pos=4),
            Action('stp', 'set_target_position',  self.set_target_position,  axis_pos=5),
            Action('sii', 'set_initial_image',    self.set_initial_image,    axis_pos=6),
            Action('sti', 'set_target_image',     self.set_target_image,     axis_pos=7),

            Action('mti', 'move_to_initial',      self.move_to_initial,      axis_pos=8),
            Action('mtt', 'move_to_target',       self.move_to_target,       axis_pos=9),
            Action('rc',  'relax_controller',     self.relax_controller,     axis_pos=10),
            Action('mm',  'mannequin_mode',       self.mannequin_mode,       axis_pos=11),
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
        
        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(4, 4)
        self._gs_action_axis                = self._gs[0:1, 0:4]
        self._gs_target_output              = self._gs[1:3, 0:2]
        self._gs_initial_image_visualizer   = self._gs[3:4, 0:1]
        self._gs_target_image_visualizer    = self._gs[3:4, 1:2]
        self._gs_action_output              = self._gs[1:2, 2:4]
        self._gs_image_visualizer           = self._gs[2:4, 2:4]

        # Create GUI components.
        self._action_axis = ActionAxis(self._fig, self._gs_action_axis, 3, 4, self._actions,
                ps3_process_rate=self._hyperparams['ps3_process_rate'],
                ps3_topic=self._hyperparams['ps3_topic'],
                ps3_button=self._hyperparams['ps3_button'],
                inverted_ps3_button=self._hyperparams['inverted_ps3_button'])
        self._target_output = OutputAxis(self._fig, self._gs_target_outputs,
                log_filename=self._log_filename)
        self._initial_image_visualizer = ImageVisualizer(self._fig, self._gs_initial_image_visualizer)
        self._target_image_visualizer = ImageVisualizer(self._fig, self._gs_target_image_visualizer)
        self._action_output = OutputAxis(self._fig, self._gs_action_output)
        self._image_visualizer = ImageVisualizer(self._hyperparams, self._fig, self._gs_image_visualizer, 
                cropsize=(240, 240), rostopic=self._hyperparams['image_topic'])

        # Setup GUI components.
        self.reload_positions()
        self.update_target_text()
        self._fig.canvas.draw()

    # Target Setup Functions.
    def prev_target_number(self, event=None):
        self._target_number = (self._target_number - 1) % self._num_targets      
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()

    def next_target_number(self, event=None):
        self._target_number = (self._target_number + 1) % self._num_targets        
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()

    def prev_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number - 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()

    def next_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number + 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()

    def set_initial_position(self, event=None):
        try:
            sample = self._agent.get_data(arm=self._actuator_type)
        except TimeoutException as e:
            self.set_action_text('set_initial_position: failure (TimeoutException while retrieving sample)')
            return
        ja = sample.get(JOINT_ANGLES)
        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)
        self._initial_position = (ja, ee_pos, ee_rot)
        save_pose_to_npz(self._target_filename, self._actuator_name, str(self._target_number),
                'initial', self._initial_position)
        self.update_target_text()
        self.set_action_text('set_initial_position: success')

    def set_target_position(self, event=None):
        try:
            sample = self._agent.get_data(arm=self._actuator_type)
        except TimeoutException as e:
            self.set_action_text('set_target_position: failure (TimeoutException while retrieving sample)')
            return
        ja = sample.get(JOINT_ANGLES)
        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)
        self._target_position = (ja, ee_pos, ee_rot)
        save_pose_to_npz(self._target_filename, self._actuator_name, str(self._target_number),
                'target', self._target_position)

        self.update_target_text()
        self.set_action_text('set_target_position: success')

    def set_initial_image(self, event=None):
        self._initial_image = self._visualizer.get_current_image()
        if self._initial_image is None:
            self.set_action_text('set_initial_image: failure (no image available)')
            return
        save_data_to_npz(self._target_filename, self._actuator_name, str(self._target_number), 
                'initial', 'image', self._initial_image)

        self.update_target_text()
        self.set_action_text('set_initial_image: success')

    def set_target_image(self, event=None):
        self._target_image = self._visualizer.get_current_image()
        if self._target_image is None:
            self.set_action_text('set_target_image: failure (no image available)')
            return
        save_data_to_npz(self._target_filename, self._actuator_name, str(self._target_number), 
                'target', 'image', self._target_image)

        self.update_target_text()
        self.set_action_text('set_target_image: success')

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

    # GUI functions.
    def update_target_text(self):
        text = (
            'target number = %s\n' % (str(self._target_number)) +
            'actuator name = %s\n' % (str(self._actuator_name)) +
            'initial position\n\tja = %s\n\tee_pos = %s\n\tee_rot = %s\n' % self._initial_position +
            'target position \n\tja = %s\n\tee_pos = %s\n\tee_rot = %s\n' % self._target_position
        )
        self._target_output.set_text(text)

        self._initial_image_visualizer.overlay_image(self._initial_image, alpha=1.0)
        self._target_image_visualizer.overlay_image(self._target_image, alpha=1.0)
        self._image_visualizer.set_initial_image(self._initial_image, alpha=0.2)
        self._image_visualizer.set_target_image(self._target_image, alpha=0.2)

    def set_action_text(self, text=''):
        self._action_output.set_text(text)

    def reload_positions(self):
        self._initial_position = load_pose_from_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'initial')
        self._target_position  = load_pose_from_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'target')
        self._initial_image    = load_data_from_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'initial', 'image', default=None)
        self._target_image     = load_data_from_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'target',  'image', default=None)


def save_pose_to_npz(filename, actuator_name, target_number, data_time, values):
    ja, ee_pos, ee_rot = values
    save_data_to_npz(filename, actuator_name, target_number, data_time, 'ja',     ja)
    save_data_to_npz(filename, actuator_name, target_number, data_time, 'ee_pos', ee_pos)
    save_data_to_npz(filename, actuator_name, target_number, data_time, 'ee_rot', ee_rot)


def save_data_to_npz(filename, actuator_name, target_number, data_time, data_name, value):
    """
    Save data to the specified file with key (actuator_name, target_number, data_time, data_name).
    """
    key = '_'.join((actuator_name, target_number, data_time, data_name))
    save_to_npz(filename, key, value)


def save_to_npz(filename, key, value):
    """
    Save a (key,value) pair to a npz dictionary.
    Args:
        filename: The file containing the npz dictionary.
        key: The key (string).
        value: The value (numpy array).
    """
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename, **tmp)


def load_pose_from_npz(filename, actuator_name, target_number, data_time):
    ja = load_data_from_npz(filename, actuator_name, target_number, data_time, 'ja',    
            default=np.zeros(7))
    ee_pos = load_data_from_npz(filename, actuator_name, target_number, data_time, 'ee_pos',
            default=np.zeros(3))
    ee_rot = load_data_from_npz(filename, actuator_name, target_number, data_time, 'ee_rot',
            default=np.zeros((3, 3)))
    return (ja, ee_pos, ee_rot)


def load_data_from_npz(filename, actuator_name, target_number, data_time, data_name, default=None):
    """
    Load data from the specified file with key (actuator_name, target_number, data_time, data_name).
    """
    key = '_'.join((actuator_name, target_number, data_time, data_name))
    return load_from_npz(filename, key, default)


def load_from_npz(filename, key, default=None):
    """
    Load a (key,value) pair from a npz dictionary.
    Args:
        filename: The file containing the npz dictionary.
        key: The key (string).
        value: The default value to return, if key or file not found.
    """
    try:
        with np.load(filename) as f:
            return f[key]
    except (IOError, KeyError) as e:
        # print 'error loading %s from %s' % (key, filename), e
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
