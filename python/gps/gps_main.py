""" This file defines the main object that runs experiments. """

import logging
import imp
import os
import os.path
import sys
import copy
import argparse
import threading
import time

from gps.gui.target_setup_gui import TargetSetupGUI
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config):
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui'] else None

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns: None
        """
        itr_start = self._initialize(itr_load)

        for itr in range(itr_start, self._hyperparams['iterations']):
            for cond in range(self._conditions):
                for i in range(self._hyperparams['num_samples']):
                    self._draw_sample(itr, cond, i)

            self._take_iteration(itr)
            self._draw_policy_samples()
            self._log_data(itr)

        self._end()

    def _initialize(self, itr_load):
        """
        Initialize from the specified iteration.
        Args:
            itr_load: If specified, loads algorithm state from that
                iteration, and resumes training at the next iteration.
        Returns:
            itr_start: Iteration to start from.
        """
        if itr_load is not None:
            self.algorithm = self.data_logger.unpickle(
                self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr_load))
            if self.gui:
                self.gui.set_status_text(
                    'Resuming training from algorithm state at iter %02d.' %
                    itr_load)
                self.gui.update(self.algorithm, itr_load)
            return itr_load + 1
        return 0

    def _draw_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        pol = self.algorithm.cur[cond].traj_distr
        if self.gui:
            self.gui.set_status_text(
                'Sampling: iter %d, condition %d, sample %d.' % (itr, cond, i))
            redo = True
            while redo:
                while self.gui.mode in ('wait', 'request', 'process'):
                    if self.gui.mode in ('wait', 'process'):
                        time.sleep(0.01)
                        continue
                    # 'request' mode.
                    if self.gui.request == 'reset':
                        try:
                            self.agent.reset(cond)
                        except NotImplementedError:
                            self.gui.err_msg = 'Agent reset unimplemented.'
                    elif self.gui.request == 'fail':
                        self.gui.err_msg = 'Cannot fail before sampling.'
                    self.gui.process_mode()  # Complete request.

                self.agent.sample(
                    pol, cond,
                    verbose=(i < self._hyperparams['verbose_trials'])
                )

                if self.gui.mode == 'request' and self.gui.request == 'fail':
                    redo = True
                    self.gui.process_mode()
                    self.agent.delete_last_sample(cond)
                else:
                    redo = False
        else:
            self.agent.sample(
                pol, cond,
                verbose=(i < self._hyperparams['verbose_trials'])
            )

    def _take_iteration(self, itr):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Calculating.')
        sample_lists = [
            self.agent.get_samples(cond, -self._hyperparams['num_samples'])
            for cond in range(self._conditions)
        ]
        self.data_logger.pickle(
            self._data_files_dir + ('sample_itr_%02d.pkl' % itr),
            copy.copy(sample_lists)
        )
        self.algorithm.iteration(sample_lists)

    def _draw_policy_samples(self):
        """ Take samples from the policy to see how it's doing. """
        if 'verbose_policy_trials' in self._hyperparams:
            for cond in range(self._conditions):
                for _ in range(self._hyperparams['verbose_policy_trials']):
                    self.agent.sample(self.algorithm.policy_opt.policy, cond,
                                      verbose=True, save=False)

    def _log_data(self, itr):
        """
        Log data and algorithm, and update the GUI.
        Args:
            itr: Iteration number.
        Returns: None
        """
        if self.gui:
            self.gui.set_status_text('Logging data and updating GUI.')
            self.gui.update(self.algorithm, itr)
        self.data_logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(self.algorithm)
        )

    def _end(self):
        """ Finish running and exit. """
        if self.gui:
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()


def main():
    """ Main function to be run. """
    parser = argparse.ArgumentParser(description='GPSMain ArgumentParser')
    parser.add_argument('experiment', type=str,
                        help='Experiment name (and directory name).')
    parser.add_argument('-n', '--new', action='store_true',
                        help='Create new experiment.')
    parser.add_argument('-t', '--targetsetup', action='store_true',
                        help='Run target setup.')
    parser.add_argument('-r', '--resume', metavar='N', type=int,
                        help='Resume training from iter N.')
    args = parser.parse_args()

    exp_name = args.experiment
    resume_training_itr = args.resume

    exp_dir = 'experiments/' + exp_name + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'

    if args.new:
        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." %
                     (exp_name, exp_dir))
        os.makedirs(exp_dir)
        open(hyperparams_file, 'w')
        sys.exit("Experiment '%s' created.\nhyperparams file: '%s'." %
                 (exp_name, hyperparams_file))

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" %
                 (exp_name, hyperparams_file))
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if args.targetsetup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS

            agent = AgentROS(hyperparams.config['agent'])
            TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError:
            sys.exit('ROS required for target setup.')
    else:
        import random
        import numpy as np
        import matplotlib.pyplot as plt

        random.seed(0)
        np.random.seed(0)

        gps = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            run_gps = threading.Thread(
                target=lambda: gps.run(itr_load=resume_training_itr), args=()
            )
            run_gps.daemon = True
            run_gps.start()

            plt.ioff()
            plt.show()
        else:
            gps.run(itr_load=resume_training_itr)


if __name__ == "__main__":
    main()
