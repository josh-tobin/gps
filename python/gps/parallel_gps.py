""" This file allows us to run many gps experiments simultaneously"""

import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from timeit import default_timer as timer
from gps.gps_main import GPSMain
from multiprocessing import Process
import argparse
import imp
import numpy as np
import logging

def start_worker(experiment, n_exp=1):
    exp_dir = 'experiments/' + experiment + '/'
    hyperparams_file = exp_dir + 'hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    hyperparams.config['gui_on'] = False
    hyperparams.config['agent']['plot'] = False
    hyperparams.config['verbose_trials'] = 0
    hyperparams.config['verbose_policy_trials'] = 0
    hyperparams.config['algorithm']['policy_opt']['n_processes'] = n_exp
    hyperparams.config['algorithm']['save_algorithm'] = False
    hyperparams.config['algorithm']['save_policy'] = True
    gps = GPSMain(hyperparams.config)
    gps.run()

def run(experiments):
    processes = []
    n_exp = len(experiments)
    for experiment in experiments:
        try:
            p = Process(target=start_worker, args=(experiment, n_exp))
            p.start()
            processes.append(p)
        except:
            pass
    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+')
    parser.add_argument('--max_simultaneous_experiments', type=int, default=6)
    args = parser.parse_args()
    total_experiments = len(args.experiments)
    batches = int(np.ceil(float(total_experiments)/float(args.max_simultaneous_experiments)))
    print 'dividing experiments into %d batches of %d'%(batches, args.max_simultaneous_experiments)
    print 'running...'
    exp_batches = [args.experiments[i:min(total_experiments, i + args.max_simultaneous_experiments)] for i in range(0, total_experiments, args.max_simultaneous_experiments)]
    start = timer()
    for i, exp_batch in enumerate(exp_batches): 
        print 'Experiments %d - %d' %(i*args.max_simultaneous_experiments+1,
                                         i*args.max_simultaneous_experiments +
                                         len(exp_batch))
        batch_start = timer()
        run(exp_batch)
        batch_end = timer()
        print '...Time elapsed: %.2f'%(batch_end-batch_start)    
    end = timer()
    print 'Total time for %d experiments: %.2f'%(len(args.experiments),
                                               end-start)
