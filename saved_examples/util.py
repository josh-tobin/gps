'''
Simple utility classes to load the pickled algo files and plot them
on the same axes
'''
import sys
#sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
#sys.path.append('/'.join(str.split(__file__, '/')[:-2]) + '/python')
sys.path.append('python')
import numpy as np
import imp

import matplotlib
matplotlib.use('Qt4Agg')
from gps.gui.mean_plotter import MeanPlotter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cPickle as pkl
from gps.utility.data_logger import DataLogger

BASE_DIR = '/home/jt/gps/experiments/'
DATA_DIR = '/data_files/'
RESULTS_DIR = '/home/jt/gps/saved_examples/'

class ResultLoader(object):
    def __init__(self, base_dir=BASE_DIR, data_dir=DATA_DIR,
                 results_dir=RESULTS_DIR, max_itrs=10):
        self._base_dir = base_dir
        self._data_dir = data_dir
        self._results_dir = results_dir
        self._max_itrs = max_itrs

    def _load_costs_from_exp(self, experiment):
        data_logger = DataLogger()
        costs = []
        for itr in range(self._max_itrs):
            f = self._base_dir + experiment + self._data_dir +\
                    'algorithm_itr_%02d.pkl'%itr
            algo = data_logger.unpickle(f)
            costs.append([np.mean(np.sum(algo.prev[m].cs, axis=1))
                            for m in range(algo.M)])
        return costs

    def _calc_test_costs(self, experiment):
        tester = PolicyTester(experiment)
        costs = []
        for itr in range(self._max_itrs):
            costs.append(tester.test(experiment, itr, verbose=False))
        return costs

    def _dump_costs_to_file(self, experiment, costs, test=False):
        output_file = self._results_dir + experiment 
        if test:
            output_file += '_test'
        output_file += '_costs.pkl'
        with open(output_file, 'wb') as f:
            pkl.dump(costs, f)

    def _load_costs_from_file(self, experiment, test=False):
        input_file = self._results_dir + experiment 
        if test:
            input_file += '_test'
        input_file += '_costs.pkl'
        with open(input_file, 'r') as f:
            costs = pkl.load(f)
        return costs

    def load(self, experiments, include_test=True):
        if not isinstance(experiments, list):
            experiments = [experiments]
        
        costs = {}
        for experiment in experiments:
            cost_file = self._results_dir + experiment + '_costs.pkl'
            try:
                cost = self._load_costs_from_file(experiment)
            except IOError:
                cost = self._load_costs_from_exp(experiment)
                self._dump_costs_to_file(experiment, cost)
            costs[experiment] = {}
            costs[experiment]['train'] = cost

            if include_test:
                test_file = self._results_dir + experiment + '_test_costs.pkl'
                try:
                    cost = self._load_costs_from_file(experiment, test=True)
                except IOError:
                    cost = self._calc_test_costs(experiment)
                    self._dump_costs_to_file(experiment, cost, test=True)
                costs[experiment]['test'] = cost

        return costs

class ResultPlotter(object):
    def __init__(self):
        self._fig = plt.figure()
        self._gs = gridspec.GridSpec(1,1)[0]
        self._ax = plt.subplot(self._gs)
        self._colors = ['r', 'b', 'y', 'g', 'indigo', 'orange', 'peru', 
                        'grey', 'crimson', 'aqua', 'magenta', 'black']
        plt.legend(loc='best')
    
    def _plot_exp(self, exp_name, exp_data, which_exps='tt'):
        ''' Expects exp_data to be a list of lists, where each inner list
            contains the cost for each condition at a given iteration '''
        color = self._colors.pop(0)
        if which_exps == 'tt' or which_exps == 'tr':
            self._ax.plot(np.mean(exp_data['train'], axis=1), 'x', 
                          markeredgewidth=1.0, alpha=1.0, color=color)
            self._ax.plot(np.mean(exp_data['train'], axis=1), color=color,
                            label=exp_name)

        if which_exps == 'tt' or which_exps == 'te':
            self._ax.plot(np.mean(exp_data['test'], axis=1), 'o',
                          markeredgewidth=1.0, alpha=1.0, color=color)
            self._ax.plot(np.mean(exp_data['test'], axis=1), color=color, 
                                  ls='--')

    
    def plot(self, exp_dict):
        ''' Plots a dictionary of experiments on the same axis '''
        for exp_name, exp_data in exp_dict.iteritems():
            self._plot_exp(exp_name, exp_data)
        self._ax.legend()
        plt.show()


class PolicyTester(object):
    ''' Agent used for testing policies, not necessarily trained on that
        agent '''

    def __init__(self, exp_name, base_dir=BASE_DIR, data_dir=DATA_DIR):
        self._base_dir = base_dir
        self._data_dir = data_dir
        self._exp_name = exp_name
        hyperparams_file = base_dir + exp_name + '/hyperparams.py'
        hyperparams = imp.load_source('hyperparams', hyperparams_file)
        self._agent = hyperparams.config['agent']['type'](
                                            hyperparams.config['agent'])
        self._cost = hyperparams.config['algorithm']['cost']['type'](
                                    hyperparams.config['algorithm']['cost'])
        self._C = hyperparams.config['common']['conditions']
    
    @property
    def C(self): return self._C

    def _load_policy(self, policy_name, policy_iter=9):
        algo_file = self._base_dir + policy_name + self._data_dir + \
                    'algorithm_itr_%02d.pkl'%policy_iter
        print 'loading file ' + algo_file
        data_logger = DataLogger()
        algorithm = data_logger.unpickle(algo_file)
        pol = algorithm.policy_opt.policy
        return pol

    def test(self, policy_name, policy_iter, verbose=False):
        policy = self._load_policy(policy_name, policy_iter=policy_iter)
        costs = []
        for c in range(self._C):
            sample = self._agent.sample(policy, c, verbose=verbose, save=False)
            l, _, _, _, _, _ = self._cost.eval(sample)
            if verbose:
                print 'Condition %d: cost sum is %f'%(c, np.sum(l))
                #link = 'saved_examples/' + 'traj_' + policy_name + '.pkl'
                link = RESULTS_DIR + 'traj_' + self._exp_name + '.pkl'
                print link
                with open(link, 'wb') as f:
                    pkl.dump(sample.get_obs(), f)
                    pkl.dump(sample.get_U(), f)
            costs.append(np.sum(l))
        if verbose:
            print 'Total cost is %f.'%np.mean(costs)
        return costs
