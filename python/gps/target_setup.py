import logging

from gps.gui.target_setup import TargetSetup
from gps.hyperparam_pr2 import defaults as config


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
        self.ts = TargetSetup(self.agent, config['common'])
        #self.gui = GUI(self.agent, config['gui'])


if __name__ == "__main__":
    g = TargetSetup()
    # g.run()
