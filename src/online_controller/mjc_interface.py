import time
import numpy as np
import scipy
import scipy.io
import roslib
roslib.load_manifest('ddp_controller_pkg')
from sensor_msgs.msg import JointState
from ddp_controller_pkg.msg import LGPolicy, MPCState
import rospy
import logging
import cPickle
from visualization_msgs.msg import Marker
from hyperparam_defaults import defaults
from sample_data.sample_data import SampleData
from cost_state_online import CostStateTracking
from online_controller import OnlineController
from algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(suppress=True)

def get_controller(pklfile, maxT=100):
    with open(pklfile) as f:
        mat = cPickle.load(f)

    dX = mat['Dx']
    dU = mat['Du']

    # Read in mu for a CostStateOnline
    tgt = mat['cost_tgt_mu']
    wp = mat['cost_wp']
    wp.fill(0.0)
    #wp[0:7] = 1.0
    wp[14:20] = 1.0
    #wp[14:20] = 1.0
    #wp[14:21] = 0.0
    cost = CostStateTracking(wp, tgt, maxT = maxT)

    # Read in offline dynamics
    dyn_init_mu = mat['dyn_init_mu']
    dyn_init_sig = mat['dyn_init_sig']
    #dyn_init_mu, dyn_init_sig = dyndata_init()
    
    K = mat['offline_K']
    k = mat['offline_k']

    it = slice(0,33)
    ip = slice(33,59)
    Fm = (np.linalg.pinv(dyn_init_sig[it, it]).dot(dyn_init_sig[it, ip])).T
    fv = dyn_init_mu[ip] - Fm.dot(dyn_init_mu[it]);
    print Fm, fv

    # Read in dynprior sigma, mu, N, mass, logmass
    gmm = mat['gmm']
    dynprior = DynamicsPriorGMM()
    dynprior.gmm = gmm
    
    #with open('plane_contact_gmm60.pkl') as f:
    #   gmm = cPickle.load(f)

    oc = OnlineController(dX, dU, dynprior, cost, maxT = maxT, dyn_init_mu=dyn_init_mu, dyn_init_sig=dyn_init_sig, offline_K=K, offline_k=k)
    #for t in range(100):
    #   print oc.act(tgt[t,:dX], None, t, None)
    return oc

def setup_agent(T=100):
    defaults['sample_data']['T'] = T
    sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
    agent = defaults['agent']['type'](defaults['agent'], sample_data)
    return sample_data, agent

def run_lqr():
    sample_data, agent = setup_agent()
    algorithm = defaults['algorithm']['type'](defaults['algorithm'], sample_data)
    conditions = 1
    idxs = [[] for _ in range(conditions)]
    for itr in range(4):
        for m in range(conditions):
            for i in range(5):
                n = sample_data.num_samples()
                pol = algorithm.cur[m].traj_distr
                sample = agent.sample(pol, sample_data.T, m, verbose=True)
                sample_data.add_samples(sample)
                idxs[m].append(n)
        algorithm.iteration([idx[-15:] for idx in idxs])

    gmm = algorithm.prev[0].traj_info.dynamics.prior.gmm
    dX = sample_data.dX
    dU = sample_data.dU
    tgt = sample_data.get_samples(idx=[-1])[0].get_X()
    wp = np.zeros(dX)
    wp[14:20] = 1.0  #EE Points

    K = algorithm.cur[0].traj_distr.K
    k = algorithm.cur[0].traj_distr.k

    all_X = sample_data.get_X()  # N x T x dX
    all_U = sample_data.get_U()  # N x T x dX
    N, T, dX = all_X.shape
    xux_data = []
    nn_train_data=[]
    nn_train_lbl = []
    for n in range(N):
        for t in range(1,T-1):
            xux_data.append(np.concatenate([all_X[n,t,:], all_U[n,t,:], all_X[n,t+1,:]]))
            nn_train_data.append(np.concatenate([all_X[n,t,:], all_U[n,t,:]]))
            nn_train_lbl.append(all_X[n,t+1,:])
    xux_data = np.array(xux_data)
    nn_train_data = np.array(nn_train_data)
    nn_train_lbl = np.array(nn_train_lbl)
    dyn_init_mu = np.mean(xux_data, axis=0)
    dyn_init_sig = np.cov(xux_data.T)

    scipy.io.savemat('data/dyndata_mjc_air.mat', {'data': nn_train_data, 'label': nn_train_lbl})
    """
    with open('/home/justin/RobotRL/test/onlinecont_py.pkl', 'w') as f:
        mat = cPickle.dump({
                'dyn_init_mu': dyn_init_mu,
                'dyn_init_sig': dyn_init_sig,
                'cost_tgt_mu': tgt,
                'cost_wp': wp,
                'Dx': dX,
                'Du': dU,
                'gmm': gmm,
                'offline_K': K,
                'offline_k': k
            }, f)
    """

def main():
    T = 400
    sample_data, agent = setup_agent(T=T)
    controller = get_controller('/home/justin/RobotRL/test/onlinecont_py.pkl', maxT=T)
    #sample = agent.sample(controller, controller.maxT, 0, screenshot_prefix='ss/mjc_relu_noupdate/img')
    sample = agent.sample(controller, controller.maxT, 0)
    l = controller.cost.eval(sample.get_X(), sample.get_U(),0)[0]
    import pdb; pdb.set_trace()


def remap_lbl():
    mat = scipy.io.loadmat('data/dyndata_mjc_test.mat')
    scipy.io.savemat('data/dyndata_mjc_test.mat', {'data': mat['data'], 'label': mat['lbl']})

if __name__ == "__main__":
    #run_lqr()
    main()
