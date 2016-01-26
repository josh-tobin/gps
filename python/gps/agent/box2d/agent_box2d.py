from copy import deepcopy
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import agent_box2d
from gps.proto.gps_pb2 import *
from gps.sample.sample import Sample

from point_mass_world import PointMassWorld


class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(agent_box2d)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()
        self._setup_world(hyperparams["world"], hyperparams["target_state"])

    def _setup_conditions(self):
        def setup(value, n):
            if not isinstance(value, list):
                try:
                    return [value.copy() for _ in range(n)]
                except AttributeError:
                    return [value for _ in range(n)]
            assert len(value) == n, 'number of elements must match number of conditions'
            return value

        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset', \
                'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)
    def _setup_world(self, world, target):
        """
        Helper method for handling setup of the Box2D world.

        """
        self.x0 = self._hyperparams["x0"]
        x0 = self._hyperparams['x0'][0]
        self._world = world(x0, target)
        self._world.run()



    def sample(self, policy, condition, verbose=True, save=True):
        """
        Runs a trial and constructs a new sample containing information about the trial.

        Args:
            policy: policy to to used in the trial
            condition (int): Which condition setup to run.
            verbose (boolean): whether or not to plot the trial
        """
        self._world.reset_world()
        b2d_X = self._world.get_state()
        new_sample = self._init_sample(b2d_X, condition)
        U = np.zeros([self.T, self.dU])
        noise = generate_noise(self.T, self.dU, smooth=self._hyperparams['smooth_noise'], \
                var=self._hyperparams['smooth_noise_var'], \
                renorm=self._hyperparams['smooth_noise_renormalize'])
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            b2d_U = policy.act(X_t, obs_t, t, noise[t,:])
            U[t,:] = b2d_U
            if (t+1) < self.T:
                for step in range(self._hyperparams['substeps']):
                     self._world.run_next(b2d_U)
                b2d_X = self._world.get_state()
                self._set_sample(new_sample, b2d_X, t, condition)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)


    def _init_sample(self, b2d_X, condition):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1, condition)
        return sample

    def _set_sample(self, sample, b2d_X, t, condition):
        for x in b2d_X.keys():
            sample.set(x, np.array(b2d_X[x]),t=t+1)
