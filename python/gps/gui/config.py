"""Default configuration and hyperparameter values for gui objects

"""
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM

""" TargetSetup """
target_setup = {
    'ps3_controller_topic': 'PS3',
    'keyboard_bindings' : {},
    'ps3_controller_bindings': {},
    'actuator_names': [TRIAL_ARM, AUXILIARY_ARM]
}

# TO-DO: hide this somewhere
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

# TO-DO: hide this somewhere
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

# TO-DO: hide these as default bindings and provide sample code for user to create custom bindings
keyboard_bindings = {
    # Target Setup
    'ptn': 'left',
    'ntn': 'right',
    'pat': 'down',
    'nat': 'up',

    'spi': 'j',
    'spt': 'k',
    'sfi': 'l',
    'sft': ';',

    'mpi': 'u',
    'mpt': 'i',
    'rc':  'o',
    'mm':  'p',

    # Training Handler
    'stop':  's',
    'st-re': 'd',
    'reset': 'f',
    'start': 'g',
}

# TO-DO: hide these as default bindings and provide sample code for user to create custom bindings
ps3 = ps3_controller_buttons    # using shorter name
ps3_controller_bindings = {
    # Target Setup
    'ptn': (ps3['r_tri_1'], ps3['l_but_l']),
    'ntn': (ps3['r_tri_1'], ps3['l_but_r']),
    'pat': (ps3['r_tri_1'], ps3['l_but_d']),
    'nat': (ps3['r_tri_1'], ps3['l_but_u']),

    'spi': (ps3['r_tri_1'], ps3['r_but_l']),
    'spt': (ps3['r_tri_1'], ps3['r_but_r']),
    'sfi': (ps3['r_tri_1'], ps3['r_but_d']),
    'sft': (ps3['r_tri_1'], ps3['r_but_u']),

    'mpi': (ps3['r_tri_2'], ps3['l_but_l']),
    'mpt': (ps3['r_tri_2'], ps3['l_but_r']),
    'rc':  (ps3['r_tri_2'], ps3['l_but_d']),
    'mm':  (ps3['r_tri_2'], ps3['l_but_u']),

    # Training Handler
    'stop':  (ps3['r_tri_2'], ps3['r_but_l']),
    'st-re': (ps3['r_tri_2'], ps3['r_but_d']),
    'reset': (ps3['r_tri_2'], ps3['r_but_u']),
    'start': (ps3['r_tri_2'], ps3['r_but_r']),
}
