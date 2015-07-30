import scipy.io
import logging
import numpy as np

from algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from hyperparam_defaults import defaults
from sample_data.sample_data import SampleData


def get_controller(matfile):
    # Need to read out a controller from matlab
    # K = controller.k
    # k = controller.U - permute(sum(bsxfun(@times,K(:,:,:),permute(X,[3 1 2])),2),[1 3 2])
    # etc.
	mat = scipy.io.loadmat(matfile)
	K = mat['K'].transpose([2,0,1])
	k = mat['k'].T
	PSig = mat['PSig'].transpose([2,0,1])
	invPSig = mat['invPSig'].transpose([2,0,1])
	cholPSig = mat['cholPSig'].transpose([2,0,1])
	return LinearGaussianPolicy(K, k, PSig, cholPSig, invPSig)

def poop_policy():
	class Policy():
		def act(self, x, obs, t, noise=None):
			return np.array([5.0,-5.0,0,0,0,0,0])
	return Policy()

def setup_agent():
    _hyperparams = defaults
    _iterations = defaults['iterations']

    sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
    agent = defaults['agent']['type'](defaults['agent'], sample_data)
    return agent

def run():
	agent = setup_agent()
	policy = get_controller('/home/justin/RobotRL/test/controller.mat')
	#policy = poop_policy()
	agent.sample(policy, 100)
	agent.sample(policy, 100)
	agent.sample(policy, 100)

run()
