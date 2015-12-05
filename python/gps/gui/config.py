"""Default configuration and hyperparameter values for gui objects

"""
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

# PS3 Joystick Buttons and Axes (documentation: http://wiki.ros.org/ps3joy)
# Mappings from ps3 controller buttons to their corresponding array indices
ps3_button = {
    'select'            :  0,
    'stick_left'        :  1,
    'stick_right'       :  2,
    'start'             :  3,
    'cross_up'          :  4,
    'cross_right'       :  5,
    'cross_down'        :  6,
    'cross_left'        :  7,
    'rear_left_2'       :  8,
    'rear_right_2'      :  9,
    'rear_left_1'       : 10,
    'rear_right_1'      : 11,
    'action_triangle'   : 12,
    'action_circle'     : 13,
    'action_cross'      : 14,
    'action_square'     : 15,
    'pairing'           : 16,
}
inverted_ps3_button = {value: key for key, value in ps3_button.iteritems()}

# Mappings from ps3 controller axes to their corresponding array indices
ps3_axis = {
    'stick_left_leftwards'      :  0,
    'stick_left_upwards'        :  1,
    'stick_right_leftwards'     :  2,
    'stick_right_upwards'       :  3,
    'button_cross_up'           :  4,
    'button_cross_right'        :  5,
    'button_cross_down'         :  6,
    'button_cross_left'         :  7,
    'button_rear_left_2'        :  8,
    'button_rear_right_2'       :  9,
    'button_rear_left_1'        : 10,
    'button_rear_right_1'       : 11,
    'button_action_triangle'    : 12,
    'button_action_circle'      : 13,
    'button_action_cross'       : 14,
    'button_action_square'      : 15,
    'acceleratometer_left'      : 16,
    'acceleratometer_forward'   : 17,
    'acceleratometer_up'        : 18,
    'gyro_yaw'                  : 19,
}
inverted_ps3_axis = {value: key for key, value in ps3_axis.iteritems()}

# Mappings from actions to their corresponding keyboard bindings
keyboard_bindings = {
    # Target Setup
    'ptn': 'left',
    'ntn': 'right',
    'pat': 'down',
    'nat': 'up',

    'sip': 'j',
    'stp': 'k',
    'sif': 'l',
    'stf': ';',

    'mti': 'u',
    'mtt': 'i',
    'rc':  'o',
    'mm':  'p',

    # Training Handler
    'stop':  's',
    'st-re': 'd',
    'reset': 'f',
    'start': 'g',
}

# Mappings from actions to their corresponding ps3 controller bindings
ps3_controller_bindings = {
    # Target Setup
    'ptn'  : (ps3_button['rear_right_1'], ps3_button['cross_left']),
    'ntn'  : (ps3_button['rear_right_1'], ps3_button['cross_right']),
    'pat'  : (ps3_button['rear_right_1'], ps3_button['cross_down']),
    'nat'  : (ps3_button['rear_right_1'], ps3_button['cross_up']),

    'sip'  : (ps3_button['rear_right_1'], ps3_button['action_square']),
    'stp'  : (ps3_button['rear_right_1'], ps3_button['action_circle']),
    'sif'  : (ps3_button['rear_right_1'], ps3_button['action_cross']),
    'stf'  : (ps3_button['rear_right_1'], ps3_button['action_triangle']),

    'mti'  : (ps3_button['rear_right_2'], ps3_button['cross_left']),
    'mtt'  : (ps3_button['rear_right_2'], ps3_button['cross_right']),
    'rc'   : (ps3_button['rear_right_2'], ps3_button['cross_down']),
    'mm'   : (ps3_button['rear_right_2'], ps3_button['cross_up']),

    # Training Handler
    'stop' : (ps3_button['rear_right_2'], ps3_button['action_square']),
    'st-re': (ps3_button['rear_right_2'], ps3_button['action_cross']),
    'reset': (ps3_button['rear_right_2'], ps3_button['action_triangle']),
    'start': (ps3_button['rear_right_2'], ps3_button['action_circle']),
}

gui = {
    'keyboard_bindings' : keyboard_bindings,
    'ps3_controller_bindings': ps3_controller_bindings,
    'ps3_controller_topic': 'joy',
    'ps3_controller_message_rate': 20,  # only process every 1 of 20 ps3 controller messages
    'actions_log_filename': 'actions_log.txt',
}

target_setup = {
    'num_targets': 10,
    'actuator_types': [TRIAL_ARM, AUXILIARY_ARM],
    'actuator_names': ['trial_arm', 'auxiliary_arm'],
}
target_setup['num_actuators'] = len(target_setup['actuator_types'])

training_handler = {
    
}