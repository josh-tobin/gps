import mjcpy2
import numpy as np

from copy import deepcopy 
from agent.agent import Agent
from agent.config import agent_mujoco
from agent.agent_util import filter_sequence


class AgentMuJoCo(Agent):
    """
    """
    def __init__(self, hyperparams, sample_data):
        config = deepcopy(agent_mujoco)
        config.update(hyperparams)
        Agent.__init__(self, config, sample_data)
        self._setup_world(hyperparams['filename'])
        #TODO: add various other parameters, e.g. for drawing and appending to state

    def _setup_world(self, filename):
        self._world = mjcpy2.MJCWorld2(filename)
        self._model = self._world.GetModel()
        self._data = self._world.GetData()
        self._options = self._world.GetOption()
        #TODO: what else goes here?

    def sample(self, policy, T, verbose=True):
        new_sample = self._init_sample()  # create new sample, populate first time step
        mj_X = new_sample.get_X(t=0)
        noise = np.random.randn(T, self.sample_data.dU)
        if self._hyperparams['smooth_noise']:
            noise = filter_sequence(noise, self._hyperparams['smooth_noise_var'], 1e-2)
            if self._hyperparams['smooth_noise_renormalize']:
                noise = noise * np.std(noise, axis=0)
        if np.any(self._hyperparams['x0var'] > 0):
            x0n = self._hyperparams['x0var'] * np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        #TODO: add noise to body pos and then setmodel
        for t in range(T):
            mj_U = policy.act(self, mj_X, new_sample.get_obs(t), t, noise[t,:])
            if verbose:
                self._world.Plot(mj_X)
            if (t+1) < T:
                if t < self._hyperparams['frozen_steps']:
                    mj_X, _ = self._world.Step(mj_X, np.zeros(self.sample_data.dU))
                else:
                    mj_X, _ = self._world.Step(mj_X, mj_U)
                #TODO: update hidden state
                self._data = self._world.GetData()
                new_sample.set('JointAngles', mj_X[:self._model['nq']], t=t+1)
                new_sample.set('JointVelocities', mj_X[self._model['nq']:], t=t+1)
                curr_eepts = self._data['site_xpos'].flatten()
                new_sample.set('EndEffectorPoints', curr_eepts, t=t+1)
                #TODO: how to set Jacobians?
                prev_eepts = new_sample.get('EndEffectorPoints', t=t)
                eept_vels = (curr_eepts - prev_eepts) / self._hyperparams['dt']
                new_sample.set('EndEffectorPointVelocities', eept_vels, t=t+1)
        #TODO: reset world
        return new_sample

    def _init_sample(self):
        sample = self.sample_data.create_new()
        #TODO: set first time step with x0, for now do something else since setmodel doesn't exist
        sample.set('JointAngles', self._model['qpos0'].flatten(), t=0)
        sample.set('JointVelocities', np.zeros(self._model['nv']), t=0)
        sites = self._data['site_xpos'].flatten()
        sample.set('EndEffectorPoints', sites, t=0)
        sample.set('EndEffectorPointVelocities', np.zeros(sites.shape), t=0)
        #TODO: how to set Jacobians?
        return sample

    def reset(self, condition):
        pass #TODO: implement setmodel
