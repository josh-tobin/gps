import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np
import imp
import argparse
import matplotlib
matplotlib.use('Qt4Agg')

from gps.gui.mean_plotter import MeanPlotter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cPickle as pkl
from gps.utility.data_logger import DataLogger

if __name__ == '__main__':
    # Set up mean plotter
    #plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(1,1)
    mp = MeanPlotter(fig, gs[0])

    # Load trajectory information
    data_logger = DataLogger()
    exp = 'pr2_mjc_scalegain'
    itrs = 10

    base_dir = '/home/jt/gps/experiments/'
    data_dir = '/data_files/'
    
    all_costs = []
    for itr in range(itrs):
        f = base_dir + exp + data_dir + 'algorithm_itr_%02d.pkl'%itr

        algo = data_logger.unpickle(f)
    
        # Plot
        costs = [np.mean(np.sum(algo.prev[m].cs, axis=1)) for m in range(algo.M)]
        all_costs.append(costs)
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                        wspace=0, hspace=0)
        fig.canvas.draw()
        fig.canvas.flush_events()
        mp.update(costs)
    with open('saved_examples/' + exp, 'wb') as of:
        pkl.dump(all_costs, of)
    plt.show()

def reload_data
