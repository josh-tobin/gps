import rospy
import roslib; roslib.load_manifest('gps_agent_pkg')
from sensor_msgs.msg import Joy

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import itertools

class ActionAxis:

    def __init__(self, actions, axarr, ps3_process_rate=20, ps3_topic='joy'):
        """
        Constructs an ActionAxis assuming actions is a dictionary of fully initialized actions:
        each action must have: key, name, func
        each action can  have: axis_pos, keyboard_binding, ps3_binding
        """
        self._actions = actions
        self._axarr = axarr
        self._fig = axarr[0].get_figure()

        # Mouse Input
        self._buttons = {}
        for key, action in self._actions.iteritems():
            if action._axis_pos is not None:
                self._buttons[key] = Button(self._axarr[action._axis_pos], '%s\n%s\n%s' % (action._name, action._kb, str(action._pb)))
                self._buttons[key].on_clicked(action._func)

        # Keyboard Input
        self._keyboard_bindings = {}
        for key, action in self._actions.iteritems():
            if action._kb is not None:
                self._keyboard_bindings[action._kb] = key
        self._cid = self._fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # PS3 Input
        self._ps3_bindings = {}
        for key, action in self._actions.iteritems():
            if action._pb is not None:
                self._ps3_bindings[action._pb] = key
        for key, value in list(self._ps3_bindings.iteritems()):
            for permuted_key in itertools.permutations(key, len(key)):
                self._ps3_bindings[permuted_key] = value
        self._ps3_count = 0
        self._ps3_process_rate = ps3_process_rate
        rospy.Subscriber(ps3_topic, Joy, self.ps3_callback)

    def on_key_press(self, event):
        if event.key in self._keyboard_bindings:
            self._actions[self._keyboard_bindings[event.key]]._func()
        else:
            print('unrecognized keyboard input: ' + str(event.key))

    def ps3_callback(self, joy_msg):
        self._ps3_count += 1
        if self._ps3_count % self._ps3_process_rate != 0:
            return
        buttons_pressed = tuple([i for i in range(len(joy_msg.buttons)) if joy_msg.buttons[i]])
        if buttons_pressed in self._ps3_bindings:
            self._actions[self._ps3_bindings[buttons_pressed]]._func()
        else:
            if not (len(buttons_pressed) == 0 or (len(buttons_pressed) == 1 and
                    (buttons_pressed[0] == ps3_button['rear_right_1'] or buttons_pressed[0] == ps3_button['rear_right_2']))):
                print('unrecognized ps3 controller input: ' + '\n' + str([inverted_ps3_button[b] for b in buttons_pressed]))

if __name__ == "__main__":
    import matplotlib.gridspec as gridspec

    from gps.gui.config import common as gui_config_common
    from gps.gui.action import Action
    
    number = 0
    def plus_1(event=None):
        global number
        number = number + 1
    def plus_2(event=None):
        global number
        number = number + 2
    def print_number(event=None):
        global number
        print(number)

    actions_arr = [
        Action('print', 'print',    print_number,   axis_pos=0, keyboard_binding='p',   ps3_binding=None),
        Action('plus1', 'plus1',    plus_1,         axis_pos=1, keyboard_binding='1',   ps3_binding=None),
        Action('plus2', 'plus2',    plus_2,         axis_pos=2, keyboard_binding='2',   ps3_binding=None),
    ]
    actions = {action._key: action for action in actions_arr}

    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(1, 1)
    gs_action = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0])
    axarr_action = [plt.subplot(gs_action[i]) for i in range(1*3)]

    ps3_process_rate = gui_config_common['ps3_process_rate']
    ps3_topic = gui_config_common['ps3_topic']

    action_axis = ActionAxis(actions, axarr_action, ps3_process_rate, ps3_topic)

    plt.ioff()
    plt.show()
