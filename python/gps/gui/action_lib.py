from gps.gui.config import keyboard_bindings, ps3_bindings

class Action:
    def __init__(self, key, name, func, keyboard_binding=None, ps3_binding=None):
        self._key = key
        self._name = name
        self._func = func
        self._kb = keyboard_binding
        self._pb = ps3_binding

target_setup_actions = [
    Action('ptn',   'prev_target_number',       self._ts.prev_target_number),
    Action('ntn',   'next_target_number',       self._ts.next_target_number),
    Action('pat',   'prev_actuator_type',       self._ts.prev_actuator_type),
    Action('nat',   'next_actuator_type',       self._ts.next_actuator_type),

    Action('sip',   'set_initial_position',     self._ts.set_initial_position),
    Action('stp',   'set_target_position',      self._ts.set_target_position),
    Action('sif',   'set_initial_features',     self._ts.set_initial_features),
    Action('stf',   'set_target_features',      self._ts.set_target_features),

    Action('mti',   'move_to_initial',          self._ts.move_to_initial),
    Action('mtt',   'move_to_target',           self._ts.move_to_target),
    Action('rc',    'relax_controller',         self._ts.relax_controller),
    Action('mm',    'mannequin_mode',           self._ts.mannequin_mode),
]
target_setup_actions = {action._key: action for action in target_setup_actions}
for key, action in target_setup_actions.iteritems():
    if key in keyboard_bindings:
        action._kb = keyboard_bindings[key]
for key, action in target_setup_actions.iteritems():
    if key in ps3_bindings:
        action._pb = ps3_bindings[key]

gps_training_actions = [
    Action('stop',  'stop',                     self._th.stop),
    Action('st-re', 'stop_reset',               self._th.stop_reset),
    Action('reset', 'reset',                    self._th.reset),
    Action('start', 'start',                    self._th.start),
]
gps_training_actions  = {action._key: action for action in gps_training_actions}
for key, action in gps_training_actions.iteritems():
    if key in keyboard_bindings:
        action._kb = keyboard_bindings[key]
for key, action in gps_training_actions.iteritems():
    if key in ps3_bindings:
        action._pb = ps3_bindings[key]