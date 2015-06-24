import mjcpy2
import numpy as np

from copy import deepcopy

from agent.agent import Agent
from agent.config import agent_mujoco


class AgentMuJoCo(Agent):
    """
    """
    def __init__(self, hyperparams, sample_data, state_assembler):
        config = deepcopy(agent_mujoco)
        config.update(hyperparams)
        Agent.__init__(self, config, sample_data, state_assembler)
        self._setup_world(hyperparams['filename'])
        self._hyperparams['dQ'] = self._model['nq']
        self._hyperparams['dV'] = self._model['nv']
        self._hyperparams['dX'] = self._model['nq'] + self._model['nv'] + self._hyperparams['dH']
        self._hyperparams['dU'] = self._model['nu'] + self._hyperparams['dH']
        self._hyperparams['dP'] = self._hyperparams['dX'] + \
                self._hyperparams['extra_phi_mean'].size  #TODO: how to actually determine dP?
        self._hyperparams['x0'] = np.concatenate([self._model['qpos0'].flatten(), \
                np.zeros(self._model['nv'])])  #TODO: implement setmodel
        #TODO: add various other parameters, e.g. for drawing and appending to state

    def _setup_world(filename):
        self._world = mjcpy2.MJCWorld2(filename)
        self._model = self._world.GetModel()
        self._data = self._world.GetData()
        self._options = self._world.GetOption()
        #TODO: what else goes here?

    def sample(self, policy, T, verbose=True):
        X = np.zeros([self._hyperparams['dX'], T])
        U = np.zeros([self._hyperparams['dU'], T])
        obs = np.zeros([self._hyperparams['dP'], T])  #TODO: populate this
        noise = np.random.randn(U.shape)
        if self._hyperparams['smooth_noise']:
            noise = filter_sequence(noise, self._hyperparams['smooth_noise_sigma'], 1e-2)
            if self._hyperparams['smooth_noise_renormalize']:
                noise = (noise.T * np.std(noise, axis=1)).T
        X[:,0] = self._hyperparams['x0']
        if np.any(self._hyperparams['x0var'] > 0):
            x0n = self._hyperparams['x0var'] * np.random.randn(self._hyperparams['x0var'].shape)
            X[:,0] += x0n
        #TODO: add noise to body pos and then setmodel
        for t in range(T):
            U[:,t] = policy.act(self, X[:,t], obs[:,t], noise[:,t], t)
            if verbose:
                self._world.Plot(X[:,t])
            if (t+1) < T:
                if t < self._hyperparams['frozen_steps']:
                    X[:,t+1] = self._world.Step(X[:,t], np.zeros(self._hyperparams['dU']))
                else:
                    X[:,t+1] = self._world.Step(X[:,t], U[:,t])
                #TODO: update hidden state
                #TODO: contruct full state, i.e. append whatever is needed, and compute obs
        #TODO: reset world
        #TODO: construct and return sample from X,U,obs
