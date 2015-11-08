import rospy
import roslib; roslib.load_manifest('gps_agent_pkg')
from sensor_msgs.msg import Joy

from datetime import datetime
import copy
import itertools
import numpy as np
import os.path
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
from matplotlib.text import Text

from gps.gui.config import target_setup, keyboard_bindings, ps3_controller_bindings
#from gps.gui.gui import GUI
#from gps.gui.action import Action, ActionLib
# from gps.gui.target_setup import TargetSetup
#from gps.gui.training_handler import TrainingHandler

from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, JOINT_ANGLES, TRIAL_ARM, AUXILIARY_ARM, TASK_SPACE, JOINT_SPACE
from gps.agent.ros.agent_ros import AgentROS
from gps_agent_pkg.msg import PositionCommand
from gps.hyperparam_pr2 import defaults as agent_config

class TargetSetup:
    def __init__(self, agent, hyperparams):
        self._agent = agent
        self._hyperparams = copy.deepcopy(target_setup)
        self._hyperparams.update(hyperparams)
        self._filedir = self._hyperparams['file_dir']

        self._gui = None
        self._actuator_names = self._hyperparams['actuator_names']
        self._target_number_max = 10
        self._actuator_number_max = len(self._actuator_names)

        self._target_number = 0
        self._actuator_number = 0
        self._actuator_type = self._actuator_names[self._actuator_number]
    
    def prev_target_number(self, event=None):
        self._target_number = (self._target_number - 1) % self._target_number_max
        self._gui.set_output("prev_target_number\n" + "target number = " + str(self._target_number))
        print(self._target_number)

    def next_target_number(self, event=None):
        self._target_number = (self._target_number + 1) % self._target_number_max
        self._gui.set_output("next_target_number\n" + "target number = " + str(self._target_number))
        print(self._target_number)

    def prev_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number - 1) % self._actuator_number_max
        self._actuator_type = self._actuator_names[self._actuator_number]
        self._gui.set_output("prev_actuator_type\n" + "actuator type = " + str(self._actuator_type))
        print(self._target_number)

    def next_actuator_type(self, event=None):
        self._actuator_number = (self._actuator_number + 1) % self._actuator_number_max
        self._actuator_type = self._actuator_names[self._actuator_number]
        self._gui.set_output("next_actuator_type\n" + "actuator type = " + str(self._actuator_type))
        print(self._target_number)

    def set_position_initial(self, event=None):
        sample = self._agent.get_data(arm=self._actuator_type)
        # TODO(chelsea) make this all go in one file.
        if self._actuator_type == TRIAL_ARM:
            # Assuming that the initial arm pose will be the same for all targets.
            filename = self._filedir + 'trialarm_initial.npz'
        elif self._actuator_type == AUXILIARY_ARM:
            filename = self._filedir + 'auxiliaryarm_initial' + self._target_number + '.npz'
        else:
            print('Unknown actuator type')
            return
        np.savez(filename, x0=sample.get(JOINT_ANGLES))
        #import ipdb; ipdb.set_trace()
        self._gui.set_output("set_initial_position: " + str(sample.get(JOINT_ANGLES).T))

    def set_position_target(self, event=None):
        """
        Grabs the current end effector points and joint angles of the trial
        arm and saves to target file.
        """
        sample = self._agent.get_data(TRIAL_ARM)
        filename = self._filedir + 'target.npz'
        add_to_npz(filename, 'ee'+str(self._target_number), sample.get(END_EFFECTOR_POINTS))
        add_to_npz(filename, 'ja'+str(self._target_number), sample.get(JOINT_ANGLES))
        self._gui.set_output("set_target_position: " + str(sample.get(END_EFFECTOR_POINTS).T))

    def set_feature_initial(self, event=None):
        pass
        
    def set_feature_target(self, event=None):
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

        filename = self._filedir + 'ft' + '_target_' + self._target_number + '.npz'
        np.savez(filename, ft_points=ft_points, ft_stable=ft_stable)
        self._gui.set_output("set_ft_target: " + "\n" +
                "ft_points: " + ft_points + "\n" +
                "ft_stable: " + ft_stable)

    def move_position_initial(self, event=None):
        filename = self._filedir + self._actuator_type + '_initial_' + self._target_number + '.npz'
        with np.load(filename) as f:
            x = f['x0']
        self._agent.reset_arm(arm=self._actuator_type, mode=JOINT_SPACE, data=x)
        self._gui.set_output("move_to_initial: " + x)

    def move_position_target(self, event=None):  # TODO - need to load up joint angles from target file.
        filename = self._filedir + 'target.npz'
        with np.load(filename) as f:
            x = f[str(self._target_number)]['ja']
        self._agent.reset_arm(arm=TRIAL_ARM, mode=JOINT_SPACE, data=x)
        self._gui.set_output("move_to_target: " + x)

    def relax_controller(self, event=None):
        self._agent.relax_arm(arm=self._actuator_type)
        self._gui.set_output("relax_controller: " + self._actuator_type)

    def mannequin_mode(self, event=None):
        # TO-DO
        self._gui.set_output("mannequin_mode: " + "NOT YET IMPLEMENTED")

def add_to_npz(filename, key, value):
    """
    Helper function for adding a new (key,value) pair to a npz dictionary.

    Note: key must be a string
    """

    tmp = {}
    if os.path.exists(filename):
        f = np.load(filename)
        for k in f.keys():
            tmp[k] = f[k]
    tmp[key] = value
    np.savez(filename,**tmp)


