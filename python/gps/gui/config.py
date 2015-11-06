"""Default configuration and hyperparameter values for gui objects

"""
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

""" TargetSetup """
target_setup = {
    'ps3_controller_topic': 'PS3',
    'keyboard_keybindings' : {},
    'ps3_controller_keybindings': {},
    'actuator_names': [TRIAL_ARM, AUXILIARY_ARM]
}

keybindings = {
	'1': 'stn1',
	'2': 'stn2',
	'3': 'stn3',
	'4': 'stn4',
	'5': 'stn5',
	'6': 'stn6',
	'7': 'stn7',
	'8': 'stn8',
	'9': 'stn9',
	'0': 'stn0',
	'z': 'sst1',
	'x': 'sst2',
	'q': 'sip',
	'w': 'stp',
	'e': 'set',
	'r': 'sft',
	'u': 'mti',
	'i': 'mti',
	'o': 'rc',
	'p': 'mm',
}

controller_bindings = {
	'1': 'stn1',
	'2': 'stn2',
	'3': 'stn3',
	'4': 'stn4',
	'5': 'stn5',
	'6': 'stn6',
	'7': 'stn7',
	'8': 'stn8',
	'9': 'stn9',
	'0': 'stn0',
	'z': 'sst1',
	'x': 'sst2',
	(4, 9): 'sip',
	(5, 9): 'stp',
	(6, 9): 'set',
	(7, 9): 'sft',
	(9, 12): 'mti',
	(9, 13): 'mti',
	(9, 14): 'rc',
	(9, 15): 'mm',
}

# controller_buttons = {
# 	0: select
# 	1: left joystick center
# 	2: right joystick center
# 	3: start
# 	4: left button up
# 	5: left button right
# 	6: left button down
# 	7: left button left
# 	8: left trigger 2
# 	9: right trigger 2
# 	10: left trigger 1
# 	11: right trigger 1
# 	12: right button up
#	13: right button right
#	14: right button down
#	15: right button left
#	16: play station
# }

# controller_axes = {
# 	0: left joystick horizontal
# 	1: left joystick vertical
# 	2: right joystick horizontal
# 	3: right joystick vertical
# 	4: left button up
# 	5: left button right
# 	6: left button down
# 	7: left button left
# 	8: left trigger 2
# 	9: right trigger 2
# 	10: left trigger 1
# 	11: right trigger 1
# 	12: right button up
#	13: right button right
#	14: right button down
#	15: right button left
#	16: tilt left/right
#	17: tilt up/down
#	18: tilt normal
#	19: ???
# }
