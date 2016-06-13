import time
import os
import numpy as np
import scipy
import scipy.io
import logging
import argparse
import cPickle


logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_controller(controllerfile, condition=0, maxT=100):
    """
    Load an online controller from controllerfile
    maxT specifies the
    """
    raise NotImplementedError()


def setup_agent(T=100):
    """Returns a MuJoCo Agent"""
    raise NotImplementedError()


def run_offline(controllerfile, verbose):
    """
    Run offline controller, and save results to controllerfile
    """
    sample_data, agent = setup_agent()
    algorithm = defaults['algorithm']['type'](defaults['algorithm'], sample_data)
    conditions = 10
    idxs = [[] for _ in range(conditions)]
    for itr in range(10):  # Iterations
        for m in range(conditions):
            for i in range(20):  # Trials per iteration
                n = sample_data.num_samples()
                pol = algorithm.cur[m].traj_distr
                sample = agent.sample(pol, sample_data.T, m, verbose=verbose)
                sample_data.add_samples(sample)
                idxs[m].append(n)
        algorithm.iteration([idx[-20:] for idx in idxs])
        print 'Finished itr ', itr

    dX = sample_data.dX
    dU = sample_data.dU

    all_X = sample_data.get_X()  # N x T x dX
    all_U = sample_data.get_U()  # N x T x dX
    N, T, dX = all_X.shape
    xux_data = []
    nn_train_data = []
    nn_train_lbl = []
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

    # Randomly shuffle data
    # N = nn_train_data.shape[0]
    # perm = np.random.permutation(N)
    # nn_train_data = nn_train_data[perm]
    # nn_train_lbl = nn_train_lbl[perm]

    # Split train/test
    # ntrain = int(N*0.8)
    # nn_test_data = nn_train_data[ntrain:]
    # nn_test_lbl = nn_train_lbl[ntrain:]
    # nn_train_data = nn_train_data[:ntrain]
    # nn_train_lbl = nn_train_lbl[:ntrain]

    # Print shapes
    print 'Data shape:'
    print 'Train data:', nn_train_data.shape
    print 'Train lbl:', nn_train_lbl.shape
    # print 'Test data:', nn_test_data.shape
    # print 'Test lbl:', nn_test_lbl.shape

    scipy.io.savemat('data/dyndata_mjc.mat', {'data': nn_train_data, 'label': nn_train_lbl})
    # scipy.io.savemat('data/dyndata_mjc_test.mat', {'data': nn_test_data, 'label': nn_test_lbl})

    controllers = []
    for condition in range(conditions):
        gmm = algorithm.prev[condition].traj_info.dynamics.prior.gmm
        tgtmu = sample_data.get_samples(idx=[-1])[0].get_X()
        # tgtmu = algorithm.cur[condition].
        K = algorithm.cur[condition].traj_distr.K
        k = algorithm.cur[condition].traj_distr.k
        controller_dict = {
            'dyn_init_mu': dyn_init_mu,
            'dyn_init_sig': dyn_init_sig,
            'cost_tgt_mu': tgtmu,
            'eetgt': np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2]),
            'Dx': dX,
            'Du': dU,
            'gmm': gmm,
            'offline_K': K,
            'offline_k': k
        }
        controllers.append(controller_dict)

    with open(controllerfile, 'w') as f:
        mat = cPickle.dump(controllers, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Train/run online controller in MuJoCo')
    parser.add_argument('-t', '--train', action='store_true', default=False, help='Train an offline controller')
    parser.add_argument('-T', '--timesteps', type=int, default=100, help='Timesteps to run online controller')
    parser.add_argument('-v', '--noverbose', action='store_true', default=False, help='Disable plotting')
    parser.add_argument('-s', '--savedata', action='store_true', default=False, help='Save dynamics data after running')
    parser.add_argument('--condition', type=int, default=0, help='Condition')

    mkdirp(os.path.join(THIS_FILE_DIR, 'controller'))
    default_file = os.path.join(THIS_FILE_DIR, 'controller', 'mjc_online.pkl')
    parser.add_argument('-c', '--controllerfile', type=str, default=default_file,
                        help='Online controller filename. Controller will be saved/loaded from here')
    args = parser.parse_args()
    return args


def run_online(T, controllerfile, condition=0, verbose=True, savedata=False):
    """
    Run online controller and save sample data to train dynamics
    """
    sample_data, agent = setup_agent(T=T)
    controller = get_controller(controllerfile, condition=condition, maxT=T)
    # sample = agent.sample(controller, controller.maxT, 0, screenshot_prefix='ss/mjc_relu_noupdate/img')
    sample = agent.sample(controller, controller.maxT, condition, verbose=verbose)
    # l = controller.cost.eval(sample.get_X(), sample.get_U(),0)[0]
    if not savedata:
        return

    X = sample.get_X()
    U = sample.get_U()
    xu = np.concatenate([X[:-1, :], U[:-1, :]], axis=1)
    xnext = X[1:, :]
    clip = np.ones((T - 1, 1))
    clip[0] = 0.0

    mkdirp(os.path.join(THIS_FILE_DIR, 'data'))
    dynmat_file = os.path.join(THIS_FILE_DIR, 'data', 'dyndata_mjc_expr3.mat')
    try:
        dynmat = scipy.io.loadmat(dynmat_file)
        dynmat['data'] = np.concatenate([dynmat['data'], xu])
        dynmat['label'] = np.concatenate([dynmat['label'], xnext])
        dynmat['clip'] = np.concatenate([dynmat['clip'], clip])
        print 'New dynamics data size: ', dynmat['data'].shape
        scipy.io.savemat(dynmat_file, {'data': dynmat['data'], 'label': dynmat['label'], 'clip': dynmat['clip']})
    except IOError:
        print 'Creating new dynamics data at:', dynmat_file
        scipy.io.savemat(dynmat_file, {'data': xu, 'label': xnext, 'clip': clip})


def mkdirp(dirname):
    """ mkdir -p """
    try:
        os.mkdir(dirname)
    except OSError:
        pass


def main():
    args = parse_args()
    if args.train:
        run_offline(args.controllerfile, verbose=not args.noverbose)
    else:
        run_online(args.timesteps, args.controllerfile, condition=args.condition, verbose=not args.noverbose,
                   savedata=args.savedata)

if __name__ == "__main__":
    main()
