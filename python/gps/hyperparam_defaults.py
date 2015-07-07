from __future__ import division

import datetime
import numpy as np

from agent.mjc.agent_mjc import AgentMuJoCo
from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from algorithm.cost.cost_state import CostState
from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from algorithm.policy.lin_gauss_init import init_lqr, init_pd
from sample_data.gps_sample_types import *

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
    'dX': 55,
    'dU': 21,
    'dO': 55,
    'state_include': [JointAngles, JointVelocities],
    'obs_include': [],
    # TODO - Have sample data compute this, and instead feed in the dimensionalities of each sensor
    'state_idx': [list(range(28)), list(range(28,55))],
    'obs_idx': [],
}

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/humanoid.xml',
    'dt': 1/20,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': 1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'args': {
        'hyperparams': {
            #'init_gains': np.ones(21)*0.0001,
            #'init_acc': np.ones(21)*1
            'init_var': 0.0001
            },
        'dt': agent['dt'],
    }
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'args': {
        'hyperparams': {
            'init_gains': np.ones(21)*0.0001,
            'init_acc': np.ones(21)*1,
            'init_var': 1000
            },
        'dQ': 21
    }
}

algorithm['cost'] = {
    'type': CostState,
    'data_types' : {
        JointAngles: {
            'wp': np.ones((1, 28)),
            'desired_state': np.array([0,0,1.4403624,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #np.zeros((1, 28)),
        },
    }
}


algorithm['dynamics'] = {
    'type': DynamicsLR,
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

defaults = {
    'iterations': 10,
    'common': common,
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}
