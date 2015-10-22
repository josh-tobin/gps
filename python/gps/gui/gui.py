# from gps.gui.config import target_setup
# import copy 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider

# GUI includes:
# Target setup (responsive to keyboard, gui, and PS3 controller)
#   - set sample number, set arm
#   - set initial position, set target position
#   - set target end effector points, set target feature points
#   - relax controller, move to initial position, move to target position
# Training controller
#   - stop, stop and reset, reset, reset and go, go
# Image visualizer: real-time image and overlay of feature points, visualize hidden states?
# Data plotter: plots losses of feature points / end effector points, joint states, feature point states, etc.
#   - plots are updated by regularly reading from a topic
# Recorder: save out plotted data

class GUI:
	def __init__(self, agent, hyperparams):
		# General
		# self._agent = agent
		# config = copy.deepcopy(target_setup)
		# config.update(hyperparams)
		# self._hyperparams = config

		# Target setup
		self.target_number = 1
		self.sensor = 'right_arm'

		# GUI components
		r, c = 5, 5
		self.fig = plt.figure(figsize=(8, 8))
		self.gs  = gridspec.GridSpec(2, 1)
		self.gs0 = gridspec.GridSpecFromSubplotSpec(r, c, subplot_spec=self.gs[0])
		self.gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs[1])

		self.functions = [('set_target_number', self.set_target_number),
						  ('set_arm', self.set_arm)]
		num_functions = len(self.functions)
		self.axarr = [plt.subplot(self.gs0[i]) for i in range(num_functions)]
		self.buttons = [Button(self.axarr[i], self.functions[i][0]) for i in range(num_functions)]
		[self.buttons[i].on_clicked(self.functions[i][1]) for i in range(num_functions)]
		
		axcolor = 'lightgoldenrodyellow'
		rax = plt.axes([0.05, 0.7, 0.15, 0.15], axisbg=axcolor)
		radio = RadioButtons(rax, ('1', '2', '3'))
		def hzfunc(label):
		    tgtdict = {'1':1, '2':2, '3':3}
		    self.target_number = tgtdict['label']
		    self.set_output("target number: " + str(self.target_number))
		radio.on_clicked(hzfunc)


		self.gs10 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.gs1[0])
		self.gs11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=self.gs1[1])

		self.output_ax = plt.subplot(self.gs11[0])
		self.output_ax.set_axis_off()
		self.set_output("please set target number")
		

#     self.c1 = CheckButtons([i for i in range(1, 14)])

#     rax = plt.axes([0.05, 0.4, 0.1, 0.15])
#     check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))

#     def func(label):
#       if label == '2 Hz': l0.set_visible(not l0.get_visible())
#       elif label == '4 Hz': l1.set_visible(not l1.get_visible())
#       elif label == '6 Hz': l2.set_visible(not l2.get_visible())
#       plt.draw()
#     check.on_clicked(func)

#     plt.show()
	
	def set_output(self, text):
		self.output_ax.text(0.95, 0.01, text,
			verticalalignment='bottom', horizontalalignment='right',
			transform=self.output_ax.transAxes, color='green', fontsize=15)



	# TARGET SETUP FUNCTIONS
	def set_target_number(self, event):
		pass

	def set_arm(self, event):
		pass

	def relax_arm(self, event):
		relax(self.arm)

	def set_initial_position(self, event):
		filename = 'matfiles/' + self.arm + '_initial_' + self.target_number + '.mat'
		x0 = get_arm_state(self.arm)	# currently not implemented
		scipy.io.savemat(filename, {'x0': x0})

	def set_target_position(self, event):
		filename = 'matfiles/' + self.arm + '_target_' + self.target_number + '.mat'
		xf = get_arm_state(self.arm)	# currently not implemented
		scipy.io.savemat(filename, {'xf': xf})

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

	def move_to_initial(self, event):
		filename = 'matfiles/' + self.arm + '_initial_' + self.target_number + '.mat'
		with scipy.io.loadmat(filename) as f:
			x0 = f['x0']
		move_arm(self.arm, x0)

	def move_to_target(self, event):
		filename = 'matfiles/' + self.arm + '_target_' + self.target_number + '.mat'
		with scipy.io.loadmat(filename) as f:
			x0 = f['x0']
		move_arm(self.arm, x0)

if __name__ == "__main__":
	g = GUI(None, None)
	plt.show()
