import logging
import imp
import os, os.path
import sys
import copy
import argparse
import threading

import matplotlib.pyplot as plt
from gps.agent.ros.agent_ros import AgentROS
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.gui.target_setup_gui import TargetSetupGUI
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
        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()

        if config['gui_on']:
            self.gui = GPSTrainingGUI(self.agent, config['common'])

            def blocking_matplotlib_event_loop():
                plt.ioff()
                plt.show()

            t = threading.Thread(target=blocking_matplotlib_event_loop, args = ())
            t.daemon = True
            t.start()
        else:
            self.gui = None

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
            # for m in range(self._conditions):
            #    self.agent.sample(self.algorithm.policy_opt.policy, m, verbose=True, save=False)

            self.data_logger.pickle(self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr), copy.copy(self.algorithm))
            self.data_logger.pickle(self._data_files_dir + ('sample_itr_%02d.pkl' % itr),    copy.copy(sample_lists))
            if self.gui:
                self.gui.update(self.algorithm)

    def resume(self, itr):
        """
        Resume training from algorithm state at specified iteration.

        itr: the iteration to which the algorithm state will be set,
             then training begins at iteration (itr + 1)
        """
        self.algorithm = self.data_logger.unpickle(self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr))
        if self.gui:
            self.gui.append_text('Resuming training from algorithm state at iteration %02d.' % itr)
            self.gui.update(self.algorithm)

        self.run(itr_start=itr+1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPS Main ArgumentParser')
    parser.add_argument('experiment', type=str, help='experiment name (and directory name)')
    parser.add_argument('-t', '--targetsetup', action='store_true', help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int, help='resume training from iteration N')
    args = parser.parse_args()

    experiment_name = args.experiment
    run_target_setup = args.targetsetup
    resume_training_itr = args.resume

    hyperparams_filepath = 'experiments/' + experiment_name + '/hyperparams.py'
    if not os.path.exists(hyperparams_filepath):
        sys.exit('Invalid experiment name: \'%s\'.\nDid you create \'%s\'?' % (experiment_name, hyperparams_filepath))
    hyperparams = imp.load_source('hyperparams', hyperparams_filepath)

    if run_target_setup:
        agent = AgentROS(hyperparams.config['agent'])
        target_setup_gui = TargetSetupGUI(agent, hyperparams.config['common'])

        plt.ioff()
        plt.show()
    else:
        import random; random.seed(0)
        import numpy as np; np.random.seed(0)

        g = GPSMain(hyperparams.config)
        if resume_training_itr is None:
            g.run()
        else:
            g.resume(resume_training_itr)
