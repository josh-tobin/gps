import copy
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider

from gps.gui.config import target_setup
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, JOINT_ANGLES
from gps_agent_pkg.msg import RelaxCommand.LEFT_ARM as ARM_LEFT
from gps_agent_pkg.msg import RelaxCommand.RIGHT_ARM as ARM_RIGHT
from gps_agent_pkg.msg import PositionCommand
# GUI includes:
# Target setup (responsive to keyboard, gui, and PS3 controller)
#   - set target number, set sensor type
#	- relax controller, mannequin mode
#   - set initial position (joint angles), move to initial position
#	- set target position (joint angles), move to target position
#	- set target end effector points, set target feature points
# Training controller
#   - stop, stop and reset, reset, reset and go, go
# Image visualizer: real-time image and overlay of feature points, visualize hidden states?
# Data plotter: plots losses of feature points / end effector points, joint states, feature point states, etc.
#   - plots are updated by regularly reading from a topic
# Recorder: save out plotted data

class GUI:
  def __init__(self, agent, hyperparams)
    # General
    self._agent = agent
    self._hyperparams = copy.deepcopy(target_setup)
    self._hyperparams.update(hyperparams)
    self._filedir = self._hyperparams['file_dir']

		# Target setup
		self.target_number = 1
		self.sensor_type = 'right_arm'

		# GUI components
		r, c = 5, 5
		self.fig = plt.figure(figsize=(8, 8))
		self.gs  = gridspec.GridSpec(1, 2)

		self.gs_left   = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.gs[0])
		self.gs_button = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.gs_left[0])
		self.gs_setup  = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=self.gs_button[0])

		self.gs_right  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self.gs[1])
		self.gs_run    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.gs_right[0])
		self.gs_vis    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.gs_right[1])

		# ~~~BUTTON~~~
		# SETUP
		self.actions = [('set_target_number', self.set_target_number),
						('set_sensor_type', self.set_sensor_type),
						('relax_controller', self.relax_controller),
						('mannequin_mode', self.mannequin_mode),
						('set_initial_position', self.set_initial_position),
						('move_to_initial', self.move_to_initial),
						('set_target_position', self.set_target_position),
						('move_to_target', self.move_to_target),
						('set_ee_target', self.set_ee_target),
						('set_ft_target', self.set_ft_target)]
		num_actions = len(self.actions)
		self.axarr = [plt.subplot(self.gs_setup[i]) for i in range(num_actions)]
		self.buttons = [Button(self.axarr[i], self.actions[i][0]) for i in range(num_actions)]
		[self.buttons[i].on_clicked(self.actions[i][1]) for i in range(num_actions)]

		# rax = plt.axes([0.05, 0.7, 0.15, 0.15], axisbg='white')
		# radio = RadioButtons(rax, ('1', '2', '3'))
		# def hzfunc(label):
		#     tgtdict = {'1':1, '2':2, '3':3}
		#     self.target_number = tgtdict['label']
		#     self.set_output("target number: " + str(self.target_number))
		# radio.on_clicked(hzfunc)

		# ~~~ RUN ~~~
		self.run_ax = plt.subplot(self.gs_run[0])

		self.set_run_output("please set target number")

		# ~~~ VIS ~~~
		pass

	def set_run_output(self, text):
		self.run_ax.clear()
		self.run_ax.set_axis_off()
		self.run_ax.text(0.95, 0.01, text,
			verticalalignment='bottom', horizontalalignment='right',
			transform=self.run_ax.transAxes, color='green', fontsize=15)

	# SETUP FUNCTIONS
	def set_target_number(self, event):
		self.set_run_output("target number: " + str(self.target_number))

	def set_sensor_type(self, event):
		pass

	def relax_controller(self, event):
		pass

	def mannequin_mode(self, event):
		pass

	def set_initial_position(self, event):
		filename = self._filedir + self.arm + '_initial_' + self.target_number + '.mat'
		x0 = self._agent.get_data(self.arm, JOINT_ANGLES)	# TODO - this is specific to AgentROS...
		scipy.io.savemat(filename, {'x0': x0})

	def move_to_initial(self, event):
		filename = self._filedir + self.arm + '_initial_' + self.target_number + '.mat'
		with scipy.io.loadmat(filename) as f:
			x0 = f['x0']
		self._agent.reset_arm(self.arm, 0, x0)

	def set_target_position(self, event):
		filename = self._filedir + self.arm + '_target_' + self.target_number + '.mat'
		xf = self._agent.get_data(self.arm, END_EFFECTOR_POINTS)	# TODO - this is specific to AgentROS...
		scipy.io.savemat(filename, {'xf': xf})

	def move_to_target(self, event):
		filename = self._filedir + self.arm + '_target_' + self.target_number + '.mat'
		with scipy.io.loadmat(filename) as f:
			x0 = f['x0']
		move_arm(self.arm, x0)

	def set_ee_target(self, event):
		pass

	def set_ft_target(self, event):
		num_samples = 50
		for i in range(num_samples):
			samples.append(new_features)	# currently not implemented
			samples.append(ft_pres)
		ft_pres = (ft_pres.sum() >= 0.8 * num_samples)
		print(samples)
		# save files
		pass

if __name__ == "__main__":
	g = GUI(None, None)
	plt.show()
