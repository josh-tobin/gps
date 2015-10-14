import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
import inspect

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

class TargetSetup:
	def set_sample_number(self, event):
		pass

	def set_arm(self, event):
		pass

	def set_initial_position(self, event):
		pass

	def set_target_position(self, event):
		pass

	def relax_arm(self, event):
		pass

	def move_to_initial(self, event):
		pass

	def move_to_target(self, event):
		pass

	def set_ee_target(self, event):
		pass

	def set_ft_target(self, event):
		pass

functions = inspect.getmembers(TargetSetup(), predicate=inspect.ismethod)
num_functions = len(functions)

fig, axarr = plt.subplots(num_functions)
buttons = [None for i in range(num_functions)]
for i in range(num_functions):
	name, function = functions[i]
	buttons[i] = Button(axarr[i], name)
	buttons[i].on_clicked(function)


plt.show()