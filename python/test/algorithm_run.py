"""
Undocumented test script.

python algorithm_run.py

Also need to add gps/python/gps to PYTHONPATH
ex.
export PYTHONPATH=/home/<>/gps/python/gps
"""
from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
import numpy as np
from algorithm.cost.cost_state import CostState
from sample_data.sample import Sample
from algorithm.policy.lin_gauss_init import init_lqr
from sample_data.gps_sample_types import Action
from sample_data.sample_data import SampleData, SysOutWriter
from algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
import logging

def run():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    dX = 7
    dU = 2
    x0 = np.zeros(dX)
    dt = 0.05
    T = 10
    M = 1
    sampledata = sample_data(T, dX, dU)

    hyper = {}
    hyper['conditions'] = M
    hyper['init_traj_distr'] = [init_lqr({}, x0, dX, dU, dt, T) for _ in range(M)]
    hyper['cost'] = [CostState({
        'data_types': {
            'dummy':{
                'wp': np.ones((1, dX)),
                'desired_state': np.zeros((1,dX))
            }
        }
    }, sampledata )]

    traj_opt = TrajOptLQRPython({})
    hyper['traj_opt'] = traj_opt

    alg = AlgorithmTrajOpt(hyper, None)
    alg.iteration([sample_data(T, dX, dU, N=3)])
    alg.iteration([sample_data(T, dX, dU, N=3)])
    alg.iteration([sample_data(T, dX, dU, N=3)])

def sample_data(T, dX, dU, N=0):
    sdata = SampleData({'T':T, 'dX': dX, 'dU': dU, 'dObs': 1}, {}, SysOutWriter())
    sdata._x_data_idx = {'dummy': tuple(range(dX))}
    if N>0:
        sdata._samples = [make_dummy_sample(T, dX, dU, sdata)]
    return sdata

def make_dummy_sample(T, dX, dU, sample_data):
    X = np.random.randn(T, dX)
    U = np.random.randn(T, dU)
    sample = Sample(sample_data)
    sample._X = X
    sample._U = U
    sample._obs = np.zeros((T, 1))
    sample._data = {
        'dummy': X,
        Action: U
    }
    return sample

if __name__ == "__main__":
    run()