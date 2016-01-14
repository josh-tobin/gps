import logging
import imp
import os, os.path
import sys
import copy
import argparse
import threading
import time

from gps.gui.target_setup_gui import TargetSetupGUI
from gps.gui.gps_training_gui import GPSTrainingGUI
from gps.utility.data_logger import DataLogger


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class GPSMain():
    """
    Main class to run algorithms and experiments.
    """
    def __init__(self, config):
        self._hyperparams = config
        self._iterations = config['iterations']
        self._conditions = config['common']['conditions']
        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()
        self.gui = GPSTrainingGUI(config['common']) if config['gui_on'] else None

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self, itr_load=None):
        """
        Run training by iteratively sampling and taking an iteration step.
        Args:
            itr_load: If specified, loads algorithm state from that iteration, and resumes training
                at the next iteration.
        """
        if self.gui:
            if itr_load is not None:
                self.gui.set_status_text('Resuming training from algorithm state at iteration %02d.'
                        % itr_load)
                self.algorithm = self.data_logger.unpickle(
                        self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr_load))
                self.gui.update(self.algorithm, itr_load)
                itr_start = itr_load + 1
            else:
                itr_start = 0

            n = self._hyperparams['num_samples']
            for itr in range(itr_start, self._iterations):
                for m in range(self._conditions):
                    for i in range(n):
                        self.gui.set_status_text('Sampling: iteration %d, condition %d, sample %d.'
                                % (itr, m, i))
                        pol = self.algorithm.cur[m].traj_distr

                        redo = True
                        while redo:
                            while self.gui.mode in ('wait', 'request', 'process'):
                                if self.gui.mode in ('wait', 'process'):
                                    time.sleep(0.01)
                                else:  # 'request' mode
                                    if self.gui.request == 'reset':
                                        try:
                                            self.agent.reset(m)
                                        except NotImplementedError as e:
                                            self.gui.err_msg = 'Agent reset not implemented.'
                                    elif self.gui.request == 'fail':
                                        self.gui.err_msg = 'Cannot fail before sampling.'
                                    self.gui.process_mode()  # complete request

                            self.agent.sample(pol, m,
                                    verbose=(i < self._hyperparams['verbose_trials']))

                            if self.gui.mode == 'request' and self.gui.request == 'fail':
                                redo = True
                                self.gui.process_mode()
                            else:
                                redo = False

                self.gui.set_status_text('Calculating.')
                sample_lists = [self.agent.get_samples(m, -n) for m in range(self._conditions)]
                self.data_logger.pickle(self._data_files_dir + ('sample_itr_%02d.pkl' % itr),
                        copy.copy(sample_lists))
                self.algorithm.iteration(sample_lists)

                # Take samples from the policy to see how it is doing.
                if 'verbose_policy_trials' in self._hyperparams:
                    for m in range(self._conditions):
                        for _ in range(self._hyperparams['verbose_policy_trials']):
                            self.agent.sample(self.algorithm.policy_opt.policy, m, verbose=True,
                                    save=False)

                self.gui.set_status_text('Logging data and updating gui.')
                self.data_logger.pickle(self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
                        copy.copy(self.algorithm))
                self.gui.update(self.algorithm, itr)
            self.gui.set_status_text('Training complete.')
            self.gui.end_mode()
        else:
            if itr_load is not None:
                self.algorithm = self.data_logger.unpickle(
                        self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr_load))
                itr_start = itr_load + 1
            else:
                itr_start = 0

            n = self._hyperparams['num_samples']
            for itr in range(itr_start, self._iterations):
                for m in range(self._conditions):
                    for i in range(n):
                        pol = self.algorithm.cur[m].traj_distr
                        self.agent.sample(pol, m, verbose=(i < self._hyperparams['verbose_trials']))

                sample_lists = [self.agent.get_samples(m, -n) for m in range(self._conditions)]
                self.data_logger.pickle(self._data_files_dir + ('sample_itr_%02d.pkl' % itr),
                        copy.copy(sample_lists))
                self.algorithm.iteration(sample_lists)

                # Take samples from the policy to see how it is doing.
                if 'verbose_policy_trials' in self._hyperparams:
                    for m in range(self._conditions):
                        for _ in range(self._hyperparams['verbose_policy_trials']):
                            self.agent.sample(self.algorithm.policy_opt.policy, m, verbose=True,
                                    save=False)

                self.data_logger.pickle(self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
                        copy.copy(self.algorithm))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPS_Main ArgumentParser')
    parser.add_argument('experiment', type=str, help='experiment name (and directory name)')
    parser.add_argument('-n', '--new', action='store_true', help='create new experiment')
    parser.add_argument('-t', '--targetsetup', action='store_true', help='run target setup')
    parser.add_argument('-r', '--resume', metavar='N', type=int, help='resume training from iter N')
    args = parser.parse_args()

    exp_name = args.experiment
    run_target_setup = args.targetsetup
    new_experiment = args.new
    resume_training_itr = args.resume

    exp_dir = 'experiments/' + exp_name + '/'
    hyperparams_file =  exp_dir + 'hyperparams.py'

    if new_experiment:
        if os.path.exists(exp_dir):
            sys.exit("Experiment '%s' already exists.\nPlease remove '%s'." % (exp_name, exp_dir))
        os.makedirs(exp_dir)
        open(hyperparams_file, 'w')
        sys.exit("Experiment '%s' created.\nhyperparams file: '%s'." % (exp_name, hyperparams_file))

    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))
    hyperparams = imp.load_source('hyperparams', hyperparams_file)

    if run_target_setup:
        try:
            import matplotlib.pyplot as plt
            from gps.agent.ros.agent_ros import AgentROS

            agent = AgentROS(hyperparams.config['agent'])
            target_setup_gui = TargetSetupGUI(hyperparams.config['common'], agent)

            plt.ioff()
            plt.show()
        except ImportError as e:
            sys.exit('ROS required for target setup.')
    else:
        import random; random.seed(0)
        import numpy as np; np.random.seed(0)
        import matplotlib.pyplot as plt

        g = GPSMain(hyperparams.config)
        if hyperparams.config['gui_on']:
            t = threading.Thread(target=lambda: g.run(itr_load=resume_training_itr), args=())
            t.daemon = True
            t.start()

            plt.ioff()
            plt.show()
        else:
            g.run(itr_load=resume_training_itr)
