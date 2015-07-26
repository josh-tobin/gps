import scipy.io
import logging
import numpy as np

from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from hyperparam_defaults import defaults
from sample_data.sample_data import SampleData
from utility.gmm import GMM
from cost_state_online import CostStateTracking
from algorithm.cost.cost_utils import approx_equal
from algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from online_controller import OnlineController

def get_controller(matfile):
    # Need to read out a controller from matlab
	mat = scipy.io.loadmat(matfile)

	dX = mat['Dx']
	dU = mat['Du']

	# Read in mu for a CostStateOnline
	tgt = mat['cost_tgt_mu'].T
	wp = mat['cost_wp'][:,0]
	cost = CostStateTracking(wp, tgt)

	# Read in offline dynamics
	Fm = mat['dyn_fd'].transpose(2,0,1)
	fv = mat['dyn_fc'].T
	dynsig = mat['dyn_sig'].transpose(2,0,1)

	# Read in dynprior sigma, mu, N, mass, logmass
	gmm = GMM()
	gmm.sigma = mat['gmm_sigma']
	gmm.mu = mat['gmm_mu']
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

	oc = OnlineController(dX, dU, dynprior, cost, offline_fd = Fm, offline_fc = fv, offline_dynsig=dynsig)
	#for t in range(100):
	#	print oc.act(tgt[t,:dX], None, t, None)
	return oc

def setup_agent():
    _hyperparams = defaults
    _iterations = defaults['iterations']

    sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
    agent = defaults['agent']['type'](defaults['agent'], sample_data)
    return agent

def run():
	#agent = setup_agent()
	oc = get_controller('/home/justin/RobotRL/test/onlinecont.mat')

run()
