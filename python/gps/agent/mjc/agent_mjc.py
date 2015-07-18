import mjcpy2
import numpy as np

from copy import deepcopy 
from agent.agent import Agent
from agent.agent_utils import generate_noise
from agent.config import agent_mujoco
from sample_data.gps_sample_types import *
from scipy.ndimage.filters import gaussian_filter


class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through this class.
    """
    def __init__(self, hyperparams, sample_data):
        config = deepcopy(agent_mujoco)
        config.update(hyperparams)
        Agent.__init__(self, config, sample_data)
        self._setup_world(hyperparams['filename'])
        #TODO: add various other parameters, e.g. for drawing and appending to state

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.

        Args:
            filename: path to XML file containing the world information
        """
        self._world = mjcpy2.MJCWorld2(filename)
        options = {
            'timestep': self._hyperparams['dt']/self._hyperparams['substeps'],
            'disableflags': 0,
        }
        if self._hyperparams['rk']:
            options['integrator'] = 3  # 3 = Runge-Katta
        else:
            options['integrator'] = 2  # 2 = Semi-implicit Euler
        self._world.SetOption(options)
        self._model = self._world.GetModel()
        self._data = self._world.GetData()
        self._options = self._world.GetOption()

        self.n_joints = self._model['nq']
        self._joint_idx = [i for i in range(self.n_joints) if i not in self._hyperparams['frozen_joints']]
        self._vel_idx = [i+self.n_joints for i in self._joint_idx]

        #TODO: This is a hack to initialize end-effector sites. 
        self.init_pose = self._hyperparams['init_pose']
        x0 = np.r_[self.init_pose, np.zeros_like(self.init_pose)]
        self.init_mj_X = x0
        self._world.Plot(x0)
        self._world.Step(x0, np.zeros(self._model['nu']))

        # Initialize x0
        self._data = self._world.GetData()
        sites = self._data['site_xpos'].flatten()
        init_vel = np.zeros(len(self._vel_idx))
        init_joints = np.array([self._model['qpos0'].flatten()[i] for i in self._joint_idx])
        # TODO: Remove hardcoded indices from state
        self.x0 = np.concatenate([init_joints, init_vel, sites, np.zeros_like(sites)])

    def sample(self, policy, T, verbose=True):
        """
        Runs a trial and constructs and returns a new sample containing information
        about the trial.

        Args:
            policy: policy to to used in the trial
            T: number of time steps for the trial
            verbose: whether or not to plot the trial
        """
        new_sample = self._init_sample()  # create new sample, populate first time step
        #X = new_sample.get_X(t=0)
        mj_X = self.init_mj_X
        U = np.zeros([T, self.sample_data.dU])
        noise = generate_noise(T, self.sample_data.dU, smooth=self._hyperparams['smooth_noise'], \
                var=self._hyperparams['smooth_noise_var'], \
                renorm=self._hyperparams['smooth_noise_renormalize'])
        if np.any(self._hyperparams['x0var'] > 0):
            x0n = self._hyperparams['x0var'] * np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        #TODO: add noise to body pos and then setmodel
        for t in range(T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t,:])
            U[t,:] = mj_U
            if verbose:
                self._world.Plot(mj_X)
            if (t+1) < T:
                #import pdb; pdb.set_trace()
                mj_X, _ = self._world.Step(mj_X, mj_U)
                #TODO: update hidden state
                self._data = self._world.GetData()
                self._set_sample(new_sample, mj_X, t)
        #TODO: reset world
        new_sample.set(Action, U)
        return new_sample

    def _to_mj_X(self, sample, t):
        joint_angles = sample.get(JointAngles, t=t)
        joint_vel = sample.get(JointVelocities, t=t)
        return np.r_[joint_angles, joint_vel]

    def _init_sample(self):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = self.sample_data.create_new()
        #TODO: set first time step with x0, for now do something else since setmodel doesn't exist
        mj_X = np.r_[self._model['qpos0'].flatten(), np.zeros(self._model['nv'])]
        sample.set(JointAngles, np.array([mj_X[i] for i in self._joint_idx]), t=0)
        sample.set(JointVelocities, np.array([mj_X[i] for i in self._vel_idx]), t=0)
        sites = self._data['site_xpos'].flatten()
        sample.set(EndEffectorPoints, sites, t=0)
        sample.set(EndEffectorPointVelocities, np.zeros(sites.shape), t=0)
        #TODO: set Jacobians
        return sample

    def _set_sample(self, sample, mj_X, t):
        sample.set(JointAngles, np.array([mj_X[i] for i in self._joint_idx]), t=t+1)
        sample.set(JointVelocities, np.array([mj_X[i] for i in self._vel_idx]), t=t+1)
        curr_eepts = self._data['site_xpos'].flatten()

        sample.set(EndEffectorPoints, curr_eepts, t=t+1)
        prev_eepts = sample.get(EndEffectorPoints, t=t)
        eept_vels = (curr_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(EndEffectorPointVelocities, eept_vels, t=t+1)
        #TODO: set Jacobians

    def reset(self, condition):
        pass #TODO: implement setmodel
