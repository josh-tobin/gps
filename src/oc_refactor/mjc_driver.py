import time
import os
import numpy as np
import scipy
import scipy.io
import logging
import copy
import argparse
import cPickle
from online_controller import OnlineController
from helper import *
from gps.hyperparam_defaults import defaults, ACTION, JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, \
    END_EFFECTOR_POINT_VELOCITIES
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.sample.sample_list import SampleList
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
import common

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_controller(controllerfile, condition, cfgfiles, maxT=100):
    """
    Load an online controller from controllerfile
    maxT specifies the
    """
    with open(controllerfile+'_'+str(condition)) as f:
        controller_dict = cPickle.load(f)
        controller_dict['maxT'] = maxT
        controller_dict['condition'] = condition

    return OnlineController(cfgfiles, config_dict=controller_dict)


def setup_agent(T=100):
    """Returns a MuJoCo Agent"""
    hyperparams = copy.deepcopy(defaults['agent'])
    hyperparams['T'] = T
    hyperparams['sensor_dims'] = {ACTION: 7, JOINT_ANGLES:7, JOINT_VELOCITIES:7,
                                  END_EFFECTOR_POINTS: 6, END_EFFECTOR_POINT_VELOCITIES: 6}
    hyperparams['state_include'] = [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES]
    dX = sum([hyperparams['sensor_dims'][k] for k in hyperparams['state_include']])
    print 'dX:', dX

    hyperparams['obs_include'] = hyperparams['state_include']
    hyperparams['x0'] = np.zeros(dX-12)
    return AgentMuJoCo(hyperparams)

def setup_algorithm(agent, conditions):
    hyperparams = copy.deepcopy(defaults['algorithm'])
    hyperparams['agent'] = agent
    hyperparams['dynamics'] = {
        'type': DynamicsLRPrior,
        'prior':{
            'type': DynamicsPriorGMM
        },
        'regularization': 1e-5
    }
    hyperparams['init_traj_distr']['T'] = agent.T
    hyperparams['init_traj_distr']['dt'] = 0.05
    algorithm = defaults['algorithm']['type'](hyperparams)
    return algorithm

def run_offline(out_filename, verbose, conditions=10, alg_iters=1, sample_iters=2):
    """
    Run offline controller, and save results to controllerfile
    """
    agent = setup_agent()
    algorithm = setup_algorithm(agent, conditions)
    samples = [[] for _ in range(conditions)]
    for itr in range(alg_iters):  # Iterations
        for m in range(conditions):
            for i in range(sample_iters):  # Trials per iteration
                pol = algorithm.cur[m].traj_distr
                sample = agent.sample(pol, m, verbose=verbose)
                samples[m].append(sample)
        algorithm.iteration([SampleList(sample_list[-20:]) for sample_list in samples])
        print 'Finished itr ', itr

    dX = sample.dX
    dU = sample.dU

    all_X = [sample.get_X() for sample_list in samples for sample in sample_list]
    all_X = np.r_[all_X]  # N x T x dX
    all_U = [sample.get_U() for sample_list in samples for sample in sample_list]
    all_U = np.r_[all_U]  # N x T x dX
    N, T, dX = all_X.shape
    xux_data = []
    nn_train_data = []
    nn_train_lbl = []
    nn_train_clip = []
    for n in range(N):
        for t in range(1, T - 1):
            xux_data.append(np.concatenate([all_X[n, t, :], all_U[n, t, :], all_X[n, t + 1, :]]))
            nn_train_data.append(np.concatenate([all_X[n, t, :], all_U[n, t, :]]))
            nn_train_lbl.append(all_X[n, t + 1, :])
    xux_data = np.array(xux_data)
    nn_train_data = np.array(nn_train_data)
    nn_train_lbl = np.array(nn_train_lbl)
    clip = np.ones(nn_train_data.shape[0])
    for i in range(0, nn_train_data.shape[0], T - 1):
        clip[i] = 0

    dyn_init_mu = np.mean(xux_data, axis=0)
    dyn_init_sig = np.cov(xux_data.T)

    mkdir_p('data')
    scipy.io.savemat(os.path.join('data', common.OFFLINE_DYNAMICS_DATA), {'data': nn_train_data, 'label': nn_train_lbl, 'clip':clip})

    controllers = []
    for condition in range(conditions):
        gmm = algorithm.prev[condition].traj_info.dynamics.prior.gmm
        #tgtmu = sample_data.get_samples(idx=[-1])[0].get_X()
        # tgtmu = algorithm.cur[condition].
        K = algorithm.cur[condition].traj_distr.K
        k = algorithm.cur[condition].traj_distr.k
        controller_dict = {
            'dyn_init_mu': dyn_init_mu,
            'dyn_init_sig': dyn_init_sig,
            #'cost_tgt_mu': tgtmu,
            'eetgt': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
            'Dx': dX,
            'Du': dU,
            'gmm': gmm,
            'offline_K': K,
            'offline_k': k
        }
        with open(out_filename+'_'+str(condition), 'w') as f:
            cPickle.dump(controller_dict, f)
        controllers.append(controller_dict)


def run_online(T, controllerfile, cfgfiles, condition=0, verbose=True, savedata=None):
    """
    Run online controller and save sample data to train dynamics
    """
    agent = setup_agent(T=T)
    controller = get_controller(controllerfile, condition, cfgfiles, maxT=T)
    sample = agent.sample(controller, condition, verbose=verbose)
    if savedata is None:
        return

    X = sample.get_X()
    U = sample.get_U()
    xu = np.concatenate([X[:-1, :], U[:-1, :]], axis=1)
    xnext = X[1:, :]
    clip = np.ones((T - 1, 1))
    clip[0] = 0.0

    mkdir_p(os.path.join(THIS_FILE_DIR, 'data'))
    dynmat_file = os.path.join(THIS_FILE_DIR, 'data', savedata)
    try:
        dynmat = scipy.io.loadmat(dynmat_file)
        dynmat['data'] = np.concatenate([dynmat['data'], xu])
        dynmat['label'] = np.concatenate([dynmat['label'], xnext])
        dynmat['clip'] = np.concatenate([dynmat['clip'], clip])
        print 'Appending to existing data. New dynamics data size: ', dynmat['data'].shape
        scipy.io.savemat(dynmat_file, {'data': dynmat['data'], 'label': dynmat['label'], 'clip': dynmat['clip']})
    except IOError:
        print 'Creating new dynamics data at:', dynmat_file
        scipy.io.savemat(dynmat_file, {'data': xu, 'label': xnext, 'clip': clip})


def main():
    print 'TEMP:main'
    args = parse_args()
    if args.offline:
        run_offline(args.controllerfile, verbose=not args.noverbose)
    else:
        run_online(args.timesteps, args.controllerfile, args.config,
                   condition=args.condition, verbose=not args.noverbose,
                   savedata=args.savedata)


def parse_args():
    parser = argparse.ArgumentParser(description='Train/run online controller in MuJoCo')
    parser.add_argument('-o', '--offline', action='store_true', default=False, help='Run & train an offline controller')
    parser.add_argument('-T', '--timesteps', type=int, default=100, help='Timesteps to run online controller')
    parser.add_argument('-n', '--noverbose', action='store_true', default=False, help='Disable plotting')
    parser.add_argument('-s', '--savedata', default=None, help='Save dynamics data after running. (Filename)')
    parser.add_argument('--condition', type=int, default=0, help='Condition')

    mkdir_p(os.path.join(THIS_FILE_DIR, 'controller'))
    default_file = os.path.join(THIS_FILE_DIR, 'controller', 'mjc_controller.pkl')
    parser.add_argument('-c', '--controllerfile', type=str, default=default_file,
                        help='Online controller filename. Controller will be saved/loaded from here')

    default_cfg = ['config_basic']
    parser.add_argument('--config', type=str, nargs='+', default=default_cfg,
                        help='Online controller config files')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
