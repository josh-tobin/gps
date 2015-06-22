import mjcpy2
import numpy as np

from agent.agent import Agent
from agent.config import agent_mujoco


class AgentMuJoCo(Agent):
    """
    """
    def __init__(self, hyperparams, sample_data, state_assembler):
        config = agent_mujoco.deepcopy()
        config.update(hyperparams)
        Agent.__init__(self, config, sample_data, state_assembler)
        self._setup_world(hyperparams['filename'])

    def _setup_world(filename):
        self.world = mjcpy2.MJCWorld2(filename)
        self.model = self.world.GetModel()
        self.data = self.world.GetData()
        self.option = self.world.GetOption()
        self._hyperparams['dQ'] = self.model['nq']
        self._hyperparams['dV'] = self.model['nv']
        self._hyperparams['dX'] = self.model['nq'] + self.model['nv'] + self._hyperparams['dH']
        self._hyperparams['dU'] = self.model['nu']
        #TODO: what else goes here?

    def sample(self, policy, T):
        X = np.zeros(self._hyperparams['dX'], T)
        U = np.zeros(self._hyperparams['dU'], T)
        obs = np.zeros(self._hyperparams['dP'], T) #TODO: populate this
        X[:,0] = self._hyperparams['x0']
        for t in range(T):
            U[:,t] = policy.act(self,X[:,t],obs[:,t],t)
            if (t+1) < T:
                X[:,t+1] = self.world.Step(X[:,t],U[:,t])
        #TODO: reset world
        #TODO: construct and return sample from X,U,obs
