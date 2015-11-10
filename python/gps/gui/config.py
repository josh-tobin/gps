"""Default configuration and hyperparameter values for gui objects

"""
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

# Mappings from ps3 controller buttons to their corresponding array indices
ps3_controller_buttons = {
    'select':  0,       # select
    'l_joy_c': 1,       # left joystick center
    'r_joy_c': 2,       # right joystick center
    'start':   3,       # start
    'l_but_u': 4,       # left button up
    'l_but_r': 5,       # left button right
    'l_but_d': 6,       # left button down
    'l_but_l': 7,       # left button left
    'l_tri_2': 8,       # left trigger 2
    'r_tri_2': 9,       # right trigger 2
    'l_tri_1': 10,      # left trigger 1
    'r_tri_1': 11,      # right trigger 1
    'r_but_u': 12,      # right button up (triangle)
    'r_but_r': 13,      # right button right (circle)
    'r_but_d': 14,      # right button down (X)
    'r_but_l': 15,      # right button left (square)
    'ply_sta': 16,      # play station
}

# Mappings from ps3 controller axes to their corresponding array indices
ps3_controller_axes = {
    'l_joy_h': 0,       # left joystick horizontal
    'l_joy_v': 1,       # left joystick vertical
    'r_joy_h': 2,       # right joystick horizontal
    'r_joy_v': 3,       # right joystick vertical
    'l_but_u': 4,       # left button up
    'l_but_r': 5,       # left button right
    'l_but_d': 6,       # left button down
    'l_but_l': 7,       # left button left
    'l_tri_2': 8,       # left trigger 2
    'r_tri_2': 9,       # right trigger 2
    'l_tri_1': 10,      # left trigger 1
    'r_tri_1': 11,      # right trigger 1
    'r_but_u': 12,      # right button up (triangle)
    'r_but_r': 13,      # right button right (circle)
    'r_but_d': 14,      # right button down (X)
    'r_but_l': 15,      # right button left (square)
    'tilt_h':  16,      # tilt horizontal
    'tilt_v':  17,      # tilt vertical
    'tilt_n':  18,      # tilt normal
    '???':     19,      # unknown
}

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
    'ptn': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['l_but_l']),
    'ntn': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['l_but_r']),
    'pat': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['l_but_d']),
    'nat': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['l_but_u']),

    'sip': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['r_but_l']),
    'stp': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['r_but_r']),
    'sif': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['r_but_d']),
    'stf': (ps3_controller_buttons['r_tri_1'], ps3_controller_buttons['r_but_u']),

    'mti': (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['l_but_l']),
    'mtt': (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['l_but_r']),
    'rc':  (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['l_but_d']),
    'mm':  (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['l_but_u']),

    # Training Handler
    'stop':  (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['r_but_l']),
    'st-re': (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['r_but_d']),
    'reset': (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['r_but_u']),
    'start': (ps3_controller_buttons['r_tri_2'], ps3_controller_buttons['r_but_r']),
}

gui = {
    'keyboard_bindings' : keyboard_bindings,
    'ps3_controller_bindings': ps3_controller_bindings,
    'ps3_controller_topic': 'joy',
    'log_file_name': 'actions_log.txt',
}

target_setup = {
    'num_targets': 10,
    'actuator_types': [TRIAL_ARM, AUXILIARY_ARM],
    'actuator_names': ['trial_arm', 'auxiliary_arm'],
}
target_setup['num_actuators'] = len(target_setup['actuator_types'])

training_handler = {
    
}