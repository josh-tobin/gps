import logging

from gps.gui.gui import GUI
from gps.hyperparam_defaults import defaults as config


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class TargetSetup():
    """Main class to run algorithms and experiments.

    Parameters
    ----------
    hyperparams: nested dictionary of hyperparameters, indexed by the type
        of hyperparameter
    """
    def __init__(self):
        self._hyperparams = config
        self._iterations = config['iterations']
        self._conditions = config['common']['conditions']

        self.agent = config['agent']['type'](config['agent'])
        self.gui = GUI(self.agent, defaults['gui'])

if __name__ == "__main__":
    g = TargetSetup()
    # g.run()
