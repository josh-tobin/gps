class Action:
    def __init__(self, key, name, func, keyboard_binding=None, ps3_controller_binding=None, axis=None):
        self._key = key
        self._name = name
        self._func = func
        self._kb = keyboard_binding
        self._cb = ps3_controller_binding
        self._axis = axis

class ActionLib:
    def __init__(self, target_setup, training_handler):
        self._ts = target_setup
        self._th = training_handler
        action_arr = [
                # Target Setup
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

                # Training Handler
                Action('stop',  'stop',                     self._th.stop),
                Action('st-re', 'stop_reset',               self._th.stop_reset),
                Action('reset', 'reset',                    self._th.reset),
                Action('start', 'start',                    self._th.start),
        ]
        self._actions = {action._key: action for action in action_arr}