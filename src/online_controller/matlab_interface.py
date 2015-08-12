import scipy.io
import logging
import numpy as np
import scipy.io
import cPickle

from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from sample_data.sample_data import SampleData
from utility.gmm import GMM
from cost_state_online import CostStateTracking
from algorithm.cost.cost_utils import approx_equal
from algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from online_controller import OnlineController
from train_dyn_net import get_data

def get_controller(matfile):
    # Need to read out a controller from matlab
	mat = scipy.io.loadmat(matfile)

	dX = mat['Dx']
	dU = mat['Du']

	# Read in mu for a CostStateOnline
	tgt = mat['cost_tgt_mu'].T
	wp = mat['cost_wp'][:,0]
	wp.fill(0.0)
	#wp[0:7] = 1.0
	wp[21:30] = 1.0
	#wp[14:21] = 0.0
	cost = CostStateTracking(wp, tgt)

	# Read in offline dynamics
	Fm = mat['dyn_fd'].transpose(2,0,1)
	fv = mat['dyn_fc'].T
	dynsig = mat['dyn_sig'].transpose(2,0,1)
	#big_dyn_sig = mat['dyn_big_sig'].transpose(2,0,1)
	dyn_init_mu = mat['dyn_init_mu'][:,0]
	dyn_init_sig = mat['dyn_init_sig']
	dyn_init_mu, dyn_init_sig = dyndata_init()
	
	# Read in prev controller
	K = mat['traj_K'].transpose(2,0,1)
	k = mat['traj_k'].T

	# Read in dynprior sigma, mu, N, mass, logmass
	gmm = GMM()
	gmm.sigma = mat['gmm_sigma'].transpose(2,0,1)
	gmm.mu = mat['gmm_mu'].T
	gmm.N = mat['gmm_N']
	gmm.mass = mat['gmm_mass']
	gmm.logmass = mat['gmm_logmass']
	dynprior = DynamicsPriorGMM()
	dynprior.gmm = gmm
	mu0, phi, _, _ = dynprior.eval(dX, dU, np.zeros((1,dX+dU+dX)))
	test_mu = mat['gmm_test_mu0'][:,0]
	test_phi = mat['gmm_test_phi']
	assert approx_equal(mu0, test_mu)
	assert approx_equal(phi, test_phi)

	with open('plane_contact_gmm8.pkl') as f:
		gmm = cPickle.load(f)

	oc = OnlineController(dX, dU, dynprior, cost, dyn_init_mu=dyn_init_mu, dyn_init_sig=dyn_init_sig, offline_K=K, offline_k=k, offline_fd = Fm, offline_fc = fv, offline_dynsig=dynsig, big_dyn_sig=None)
	#for t in range(100):
	#	print oc.act(tgt[t,:dX], None, t, None)
	return oc

def dyndata_init():
    train_dat, train_lbl, _, _ = get_data(['dyndata_plane_ft'],['dyndata_plane_ft_2'])
    xux = np.c_[train_dat, train_lbl]

    mu = np.mean(xux, axis=0)
    diff = xux-mu
    sig = diff.T.dot(diff)

    it = slice(0,46)
    ip = slice(46,85)
    Fm = (np.linalg.pinv(sig[it, it]).dot(sig[it, ip])).T
    fv = mu[ip] - Fm.dot(mu[it]);
    print Fm
    return mu, sig


def newgmm():
    logging.basicConfig(level=logging.DEBUG)
    train_dat, train_lbl, _, _ = get_data(['dyndata_plane_ft'],['dyndata_plane_ft_2'])
    xux = np.c_[train_dat, train_lbl]
    print xux.shape
    gmm = GMM()
    gmm.update(xux, 8)
    with open('plane_contact_gmm8.pkl', 'w') as f:
    	cPickle.dump(gmm, f)


def run():
    oc = get_controller('/home/justin/RobotRL/test/onlinecont.mat')
    print np.max(oc.offline_K[0], axis=0)
    print oc.offline_k[0]

if __name__ == "__main__":
	newgmm()
