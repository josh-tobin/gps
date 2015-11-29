import rospy
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
        rospy.init_node('gui')
        self.agent = config['agent']['type'](config['agent'], init_node=False)
        self.gui = GUI(self.agent, config['common'])

if __name__ == "__main__":
    g = TargetSetup()
    rospy.spin()
