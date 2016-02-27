""" This file defines an agent for the Box2D simulator. """
from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_BOX2D
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample


class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(AGENT_BOX2D)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()
        self._setup_world(hyperparams["world"], 0, hyperparams["target_state"])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        # 2/25: removed x0 from fields
        for field in ('x0var', 'pos_body_idx', 'pos_body_offset', \
                'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)
        # Temporary fix to not working with box example: setup x0 hyperparam
        if not isinstance(self._hyperparams['x0'], list):
            self._hyperparams['x0'] = setup(self._hyperparams['x0'], conds)
    def _setup_world(self, world, condition, target):
        """
        Helper method for handling setup of the Box2D world.
        2/25: added condition number. Added robot_config. Updated x0 to
        include robot_config
        """
        #self.x0 = self._hyperparams["x0"]
        if "robot_config" in self._hyperparams:
            self.x0 = [np.append(x0, rbt_cfg) for x0, rbt_cfg in zip(self._hyperparams["x0"], self._hyperparams["robot_config"])]
            self.robot_config = self._hyperparams["robot_config"]
        
            self._world = world(self.x0[condition], target, self.robot_config[condition])
        else:
            self.x0 = self._hyperparams["x0"]
            self.robot_config = None        
            self._world = world(self.x0[condition], target)
        self._world.run()

    def sample(self, policy, condition, verbose=False, save=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: policy to to used in the trial
            condition (int): Which condition setup to run.
            verbose (boolean): whether or not to plot the trial (not used here)
        """
        robotconf = self.robot_config[condition] if self.robot_config is not None else None
        if robotconf is not None:
            self._setup_world(self._hyperparams["world"], condition, self._hyperparams["target_state"])
        else:
            self._world.reset_world(self.x0[condition])
        b2d_X = self._world.get_state()
        new_sample = self._init_sample(b2d_X)
        U = np.zeros([self.T, self.dU])
        noise = generate_noise(self.T, self.dU, self._hyperparams)
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t+1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._world.run_next(U[t, :])
                b2d_X = self._world.get_state()
                self._set_sample(new_sample, b2d_X, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)

    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1)
        return sample

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t=t+1)
