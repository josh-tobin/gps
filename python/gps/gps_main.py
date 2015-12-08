import logging
import imp
import os
import sys
import copy

#from gps.hyperparam_defaults import defaults as config
#from gps.hyperparam_badmm_defaults import defaults as config
# from gps.hyperparam_defaults import defaults as config
# from gps.hyperparam_pr2 import defaults as config

from gps.gui.gui import GUI
from gps.utility.data_logger import DataLogger

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GPSMain():
    """Main class to run algorithms and experiments.

    Parameters
    ----------
    hyperparams: nested dictionary of hyperparameters, indexed by the type
        of hyperparameter
    """
    def __init__(self, config):
        self._hyperparams = config
        self._iterations = config['iterations']
        self._conditions = config['common']['conditions']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger(config['common'])
        self.gui = GUI(self.agent, config['common']) if 'gui' in config else None

        # TODO: the following is a hack that doesn't even work some of the time
        #       let's think a bit about how we want to really do this
        # TODO - the following line of code is needed for agent_mjc, but not agent_ros
        config['algorithm']['init_traj_distr']['args']['x0'] = self.agent.x0[0]
        config['algorithm']['init_traj_distr']['args']['dX'] = self.agent.dX
        config['algorithm']['init_traj_distr']['args']['dU'] = self.agent.dU
        config['algorithm']['dO'] = self.agent.dO

        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_start=0):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        n = self._hyperparams['num_samples']
        for itr in range(itr_start, self._iterations):
            for m in range(self._conditions):
                for i in range(n):
                    pol = self.algorithm.cur[m].traj_distr
                    self.agent.sample(pol, m, verbose=True)
            sample_lists = [self.agent.get_samples(m, -n) for m in range(self._conditions)]
            self.algorithm.iteration(sample_lists)

            # Take samples from the policy to see how it is doing.
            #for m in range(self._conditions):
            #    self.agent.sample(self.algorithm.policy_opt.policy, m, verbose=True, save=False)

            self.data_logger.pickle(copy.copy(self.algorithm), 'algorithm', itr)
            self.data_logger.pickle(copy.copy(sample_lists), 'sample', itr)
            if self.gui:
                self.gui.update(self.algorithm)

    def resume(self, itr):
        """
        Resume from iteration specified.

        itr: the iteration to which the algorithm data will be reset to,
             then training begins at iteration (itr + 1)
        """
        self.algorithm = self.data_logger.unpickle('algorithm', itr)
        if self.gui:
            gui.update(self.algorithm)

        self.run(itr_start=itr+1)

if __name__ == "__main__":
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Runs an experiment.")
        print("Usage: %s [HYPERPARAMS_PATH]" % sys.argv[0])
        print("")
        print("HYPERPARAMS_PATH: the experiment directory where hyperparams.py is located")
        print("       (default: 'default_mjc_experiment')")
    else:
        if len(sys.argv) > 1:
            file_path = os.path.join('./experiments/',sys.argv[1])
        else:
            file_path = './experiments/default_mjc_experiment/'
        hyperparams = imp.load_source('hyperparams',
                                      os.path.join(file_path, 'hyperparams.py'))

        import random; random.seed(0)
        import numpy as np; np.random.seed(0)

        g = GPSMain(hyperparams.config)
        g.run()
