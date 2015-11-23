import copy
import os.path

import numpy as np

from gps.gui.config import target_setup as target_setup_config

from gps.proto.gps_pb2 import END_EFFECTOR_POSITIONS, END_EFFECTOR_ROTATIONS, JOINT_ANGLES, TRIAL_ARM, AUXILIARY_ARM, TASK_SPACE, JOINT_SPACE

class TargetSetup:
    def __init__(self, agent, hyperparams, gui=None):
        self._agent = agent
        self._hyperparams = copy.deepcopy(target_setup_config)
        self._hyperparams.update(hyperparams)
        self._gui = gui

        self._target_files_dir = self._hyperparams['target_files_dir']

        self._num_targets = self._hyperparams['num_targets']
        self._num_actuators = self._hyperparams['num_actuators']
        self._actuator_types = self._hyperparams['actuator_types']
        self._actuator_names = self._hyperparams['actuator_names']

        self._target_number = 0
        self._actuator_number = 0
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]

    def prev_target_number(self, event=None):
        self._target_number = (self._target_number - 1) % self._num_targets
        self.output_text('prev_target_number:' + '\n' +
                         'target number = ' + str(self._target_number))

    def next_target_number(self, event=None):
        self._target_number = (self._target_number + 1) % self._num_targets
        self.output_text('next_target_number:' + '\n' +
                         'target number = ' + str(self._target_number))

    def prev_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number - 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.output_text('prev_actuator_type:' + '\n' +
                         'actuator type = ' + str(self._actuator_type) + '\n' +
                         'actuator name = ' + str(self._actuator_name))

    def next_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number + 1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.output_text('next_actuator_type:' + '\n' +
                         'actuator type = ' + str(self._actuator_type) + '\n' +
                         'actuator name = ' + str(self._actuator_name))

    def set_initial_position(self, event=None):
        # TODO - this might not work if you've already started a trial (data might be 100xD)
        sample = self._agent.get_data(arm=self._actuator_type)
        filename = self._target_files_dir + self._actuator_name + '_initial.npz'

        ja_key = 'ja' + str(self._target_number)
        ja_value = sample.get(JOINT_ANGLES)
        add_to_npz(filename, ja_key, ja_value)

        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)

        pos_key = 'ee_pos' + str(self._target_number)
        rot_key = 'rot_pos' + str(self._target_number)
        add_to_npz(filename, pos_key, ee_pos)
        add_to_npz(filename, rot_key, ee_rot)

        self.output_text('set_initial_position:' + '\n' +
                         'filename = ' + filename + '\n' +
                         ja_key + ' = ' + str(ja_value.T) + '\n' +
                         pos_key + ' = ' + str(ee_pos) + '\n' +
                         rot_key + ' = ' + str(ee_rot))

    def set_target_position(self, event=None):
        """
        Grabs the current end effector points and joint angles of the trial
        arm and saves to target file.
        """
        sample = self._agent.get_data(arm=self._actuator_type)
        filename = self._target_files_dir + self._actuator_name + '_target.npz'

        ja_key = 'ja' + str(self._target_number)
        ja_value = sample.get(JOINT_ANGLES)
        add_to_npz(filename, ja_key, ja_value)

        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)

        pos_key = 'ee_pos' + str(self._target_number)
        rot_key = 'rot_pos' + str(self._target_number)
        add_to_npz(filename, pos_key, ee_pos)
        add_to_npz(filename, rot_key, ee_rot)

        self.output_text('set_target_position:' + '\n' +
                         'filename = ' + filename + '\n' +
                         ja_key + ' = ' + str(ja_value.T) + '\n' +
                         pos_key + ' = ' + str(ee_pos) + '\n' +
                         rot_key + ' = ' + str(ee_rot))

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

        self.output_text('set_target_features' + '\n' +
                         'filename = ' + filename + '\n' +
                         fp_key + ' = ' + str(fp_value.T) + '\n' +
                         fs_key + ' = ' + str(fs_value.T))

    def move_to_initial(self, event=None):
        filename = self._target_files_dir + self._actuator_name + '_initial.npz'
        ja_key = 'ja' + str(self._target_number)
        with np.load(filename) as f:
            ja_value = f[ja_key]
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=ja_value)
        self.output_text('move_to_initial:' + '\n' +
                         ja_key + ' = ' + str(ja_value.T))

    def move_to_target(self, event=None):
        filename = self._target_files_dir + self._actuator_name + '_target.npz'
        ja_key = 'ja' + str(self._target_number)
        with np.load(filename) as f:
            ja_value = f[ja_key]
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=ja_value)
        self.output_text('move_to_target:' + '\n' +
                         ja_key + ' = ' + str(ja_value.T))

    def relax_controller(self, event=None):
        self._agent.relax_arm(arm=self._actuator_type)
        self.output_text('relax_controller:' + '\n' +
                         'actuator type = ' + str(self._actuator_type) + '\n' +
                         'actuator name = ' + str(self._actuator_name))

    def mannequin_mode(self, event=None):
        # TO-DO
        self.output_text('mannequin_mode:' + '\n' +
                         'NOT YET IMPLEMENTED')

    def output_text(self, text):
        if self._gui:
            self._gui.set_output_text(text)
        else:
            print(text)

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
