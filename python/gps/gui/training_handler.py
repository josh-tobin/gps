import copy

from gps.gui.config import training_handler as training_handler_config

class TrainingHandler:
    def __init__(self, agent, hyperparams, gui=None):
        self._agent = agent
        self._hyperparams = copy.deepcopy(training_handler_config)
        self._hyperparams.update(hyperparams)
        self._gui = gui

    def stop(self, event=None):
        self.output_text('stop')
        self.set_bgcolor('red')
        pass

    def stop_reset(self, event=None):
        self.output_text('stop_reset')
        self.set_bgcolor('orange')
        pass

    def reset(self, event=None):
        self.output_text('reset')
        self.set_bgcolor('yellow')
        pass

    def start(self, event=None):
        self.output_text('start')
        self.set_bgcolor('green')
        pass

    def set_bgcolor(self, color):
        if self._gui:
            self._gui._ax_output.set_axis_on()
            self._gui._ax_output.set_xticks([])
            self._gui._ax_output.set_yticks([])
            self._gui._ax_output.set_axis_bgcolor(color)
            self._gui._fig.canvas.draw()

    def output_text(self, text):
        if self._gui:
            self._gui.set_output_text(text)
        else:
            print(text)