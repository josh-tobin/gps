import cPickle as pickle
import numpy as np
import argparse

DATA_LOCATION = '/home/jt/gps/experiments/'

def print_exp(exp_name, exp_data):
    print '------------------------------'
    print 'Experiment %s'%exp_name
    print
    
    if exp_data['train_error']:
        max_iter = max(exp_data['train_error'].keys())
        print 'Train error:'
        print '\t iter %d, mean: %.2f \t std: %.2f'%(
            max_iter,
            np.mean(exp_data['train_error'][max_iter]),
            np.std(exp_data['train_error'][max_iter]))
        print
    if exp_data['test_error']:
        print 'Test error:'
        for key, value in exp_data['test_error'].iteritems():
            print '\t Evaluated on %s:'%key
            print '\t ',
            for member in value:
                print "%.2f "%member,
            print
            print

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name')
    args = parser.parse_args()
    with open(DATA_LOCATION + args.experiment_name + '/data_files/error_data.pkl', 'r') as f:
        data = pickle.load(f)
    print_exp(args.experiment_name, data)

if __name__ == '__main__': 
    main()
