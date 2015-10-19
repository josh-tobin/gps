import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import inspect

from config import target_setup
import copy 

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
		config = copy.deepcopy(target_setup)
		config.update(hyperparams)
		self._hyperparams = config
		self._agent = agent




	# Chelsea TO-DO
	def relax_arm(arm):
		pass

	# ARM is either 'left_arm' or 'right_arm'
	# xf is a 14-dimensional vector (7 joint positions, 7 joint velocities)
	def move_arm(arm, xf):
		pass

	# ARM is either 'left_arm' or 'right_arm'
	# return x, a 14-dimensional vector of the current joint positions and velocities
	def get_arm_state(arm):
		return x

# TARGET SETUP FUNCTIONS

	self.arm = 'right_arm'
	self.target_number = 1

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

# Create and display gui window.
ts = TargetSetup()
functions = inspect.getmembers(ts, predicate=inspect.ismethod)
num_functions = len(functions)

fig, axarr = plt.subplots(num_functions)
buttons = [None for i in range(num_functions)]
for i in range(num_functions):
	name, function = functions[i]
	buttons[i] = Button(axarr[i], name)
	buttons[i].on_clicked(function)

rax = plt.axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))

def func(label):
    if label == '2 Hz': l0.set_visible(not l0.get_visible())
    elif label == '4 Hz': l1.set_visible(not l1.get_visible())
    elif label == '6 Hz': l2.set_visible(not l2.get_visible())
    plt.draw()
check.on_clicked(func)

plt.show()
