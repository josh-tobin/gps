from datetime import datetime
import copy
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider

# from gps.gui.config import target_setup
# from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, JOINT_ANGLES
# from gps_agent_pkg.msg import RelaxCommand.LEFT_ARM as ARM_LEFT
# from gps_agent_pkg.msg import RelaxCommand.RIGHT_ARM as ARM_RIGHT
# from gps_agent_pkg.msg import PositionCommand

# ~~~ GUI Specifications ~~~
# Target setup (responsive to keyboard, gui, and PS3 controller)
#   - set target number, set sensor type
#	- relax controller, mannequin mode
#   - set initial position (joint angles), move to initial position
#	- set target position (joint angles), move to target position
#	- set target end effector points, set target feature points
#
# Robot training
#   - stop, stop and reset, reset, reset and go, go
#
# Data visualizer
#	- algorithm training visualizations
# 	- real-time image and feature points visualization
#	- overlay of initial and target feature points
#	- visualize hidden states?
#
# Data plotter
#	- algorithm training costs
#	- losses of feature points / end effector points
#	- joint states, feature point states, etc.
#
# Data recorder
#	- save tracked data to file
#	- create movie from image visualizations

class GUI:
	def __init__(self, agent, hyperparams):
		# General
		# self._agent = agent
		# self._hyperparams = copy.deepcopy(target_setup)
		# self._hyperparams.update(hyperparams)
		# self._filedir = self._hyperparams['file_dir']

		# Target setup
		self._target_number = 1
		self._sensor_names = {1: 'right_arm', 2: 'left_arm'}
		self._sensor_type = self._sensor_names[1]
		# self._output_file = self._filedir + "gui_output_" + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M')

		# GUI components
		r, c = 5, 5
		self._fig = plt.figure(figsize=(8, 8))
		self._gs  = gridspec.GridSpec(1, 2)

		self._gs_left   = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs[0])
		self._gs_setup  = gridspec.GridSpecFromSubplotSpec(5, 2, subplot_spec=self._gs_left[0])

		self._gs_right  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=self._gs[1])
		self._gs_output = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_right[0])
		self._gs_vis    = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self._gs_right[1])

		# ~~~ SETUP ~~~
		self._actions = [('set_target_number', self.set_target_number),
						('set_sensor_type', self.set_sensor_type),
						('relax_controller', self.relax_controller),
						('mannequin_mode', self.mannequin_mode),
						('set_initial_position', self.set_initial_position),
						('move_to_initial', self.move_to_initial),
						('set_target_position', self.set_target_position),
						('move_to_target', self.move_to_target),
						('set_ee_target', self.set_ee_target),
						('set_ft_target', self.set_ft_target)]
		num_actions = len(self._actions)
		self._axarr = [plt.subplot(self._gs_setup[i]) for i in range(num_actions)]
		
		self._axarr[0].text(0, 0.75, 'set_target_number')
		self._target_slider = DiscreteSlider(self._axarr[0], '', 1, 13, valinit=1, valfmt='%d')
		self._target_slider.on_changed(self.set_target_number)
		self._axarr[1].text(1, 0.75, 'set_sensor_type')
		self._sensor_slider = DiscreteSlider(self._axarr[1], '', 1,  2, valinit=1, valfmt='%d')
		self._sensor_slider.on_changed(self.set_sensor_type)

		buttons_start = 2
		self._actions_button = self._actions[buttons_start:]
		num_buttons = len(self._actions_button)
		self._buttons = [Button(self._axarr[buttons_start+i], self._actions_button[i][0]) for i in range(num_buttons)]
		[self._buttons[i].on_clicked(self._actions_button[i][1]) for i in range(num_buttons)]

		# ~~~ OUTPUT ~~~
		self._output_ax = plt.subplot(self._gs_output[0])
		self.set_output("target number: " +  str(self._target_number) + "\n" +
				"sensor type: " + self._sensor_type)

		# ~~~ VIS ~~~
		pass

	def set_output(self, text):
		self._output_ax.clear()
		self._output_ax.set_axis_off()
		self._output_ax.text(0, 1, text,
			verticalalignment='top', horizontalalignment='left',
			transform=self._output_ax.transAxes, color='green', fontsize=12)
		self._fig.canvas.draw()
		# with open(output_file, "a") as f:
		# 	f.write(text)

	# SETUP FUNCTIONS
	def set_target_number(self, val):
		self._target_number = int(val)
		self.set_output("set_target_number: " + str(self._target_number))

	def set_sensor_type(self, val):
		self._sensor_type = self._sensor_names[int(val)]
		self.set_output("set_sensor_type: " + self._sensor_type)

	def relax_controller(self, event):
		self._agent.relax_arm(self._sensor_type)
		self.set_output("relax_controller: " + self._sensor_type)

	def mannequin_mode(self, event):
		# TO-DO
		self.set_output("mannequin_mode: " + "NOT YET IMPLEMENTED")

	def set_initial_position(self, event):
		x = self._agent.get_data(self._sensor_type, JOINT_ANGLES)	# TODO - this is specific to AgentROS...
		filename = self._filedir + self._sensor_type + '_initial_' + self._target_number + '.npz'
		np.savez(filename, x=x)
		self.set_output("set_initial_position: " + x)

	def move_to_initial(self, event):
		filename = self._filedir + self._sensor_type + '_initial_' + self._target_number + '.npz'
		with np.load(filename) as f:
			x = f['x']
		self._agent.reset_arm(self._sensor_type, 0, x)
		self.set_output("move_to_initial: " + x)

	def set_target_position(self, event):
		x = self._agent.get_data(self._sensor_type, JOINT_ANGLES)	# TODO - this is specific to AgentROS...
		filename = self._filedir + self._sensor_type + '_target_' + self._target_number + '.npz'
		np.savez(filename, x=x)
		self.set_output("set_target_position: " + x)

	def move_to_target(self, event):
		filename = self._filedir + self._sensor_type + '_target_' + self._target_number + '.npz'
		with np.load(filename) as f:
			x = f['x']
		self._agent.reset_arm(self._sensor_type, 0, x)
		self.set_output("move_to_target: " + x)

	def set_ee_target(self, event):
		x = self._agent.get_data(self._sensor_type, END_EFFECTOR_POINTS)	# TODO - this is specific to AgentROS...
		filename = self._filedir + 'ee' + '_target_' + self._target_number + '.npz'
		np.savez(filename, x=x)
		self.set_output("set_ee_target: " + x)

	def set_ft_target(self, event):
		num_samples = 50
		threshold = 0.8

		ft_points_samples = np.empty()
		ft_prsnce_samples = np.empty()
		for i in range(num_samples):
			ft_points_samples.append(self._agent.get_data(self._sensor_type, VISUAL_FEATURE_POINTS))		# currently not implemented
			ft_prsnce_samples.append(self._agent.get_data(self._sensor_type, VISUAL_FEATURE_PRESENCE))	# currently not implemented
		ft_points_mean = np.mean(ft_points)
		ft_prsnce_mean = np.mean(ft_pres)

		ft_stable = np.array(ft_prsnce_mean >= threshold, dtype=int)
		ft_points = ft_stable * ft_points_mean

		filename = self._filedir + 'ft' + '_target_' + self._target_number + '.npz'
		np.savez(filename, ft_points=ft_points, ft_stable=ft_stable)
		self.set_output("set_ft_target: " + "\n" + 
				"ft_points: " + ft_points + "\n" +
				"ft_stable: " + ft_stable)

class DiscreteSlider(Slider):
	def set_val(self, val):
		self.val = val
		discrete_val = round(val)
		self.valtext.set_text(self.valfmt % discrete_val)
		self.poly.xy[2] = discrete_val, 1
		self.poly.xy[3] = discrete_val, 0
		if self.drawon:
			self.ax.figure.canvas.draw()
		if self.eventson:
			for cid, func in self.observers.iteritems():
				func(discrete_val)

if __name__ == "__main__":
	g = GUI(None, None)
	plt.show()
