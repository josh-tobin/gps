import mjcpy
import numpy as np

from copy import deepcopy
from agent.agent import Agent
from agent.agent_utils import generate_noise
from agent.config import agent_mujoco
from proto.gps_pb2 import *
from scipy.ndimage.filters import gaussian_filter


class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through this class.
    """
    def __init__(self, hyperparams, sample_data):
        config = deepcopy(agent_mujoco)
        config.update(hyperparams)
        Agent.__init__(self, config, sample_data)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

    def _setup_conditions(self):
        def setup(value, n):
            if not isinstance(value, list):
                try:
                    return [value.copy() for _ in range(n)]
                except AttributeError:
                    return [value for _ in range(n)]
            assert len(value) == n, 'number of elements must match number of conditions'
            return value

        #TODO: discuss a different way to organize some of this
        conds = self._hyperparams['conditions']
        for field in ('init_pose', 'x0var', 'pos_body_idx', 'pos_body_offset', \
                'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.

        Args:
            filename: path to XML file containing the world information
        """
        self._world = mjcpy.MJCWorld(filename)
        options = {
            'timestep': self._hyperparams['dt']/self._hyperparams['substeps'],
            'disableflags': 0,
        }
        if self._hyperparams['rk']:
            options['integrator'] = 3  # Runge-Kutta
        else:
            options['integrator'] = 2  # Semi-implicit Euler
        self._world.set_option(options)

        self._model = self._world.get_model()
        self._model = [deepcopy(self._model) for _ in range(self._hyperparams['conditions'])]
        self._options = self._world.get_option()

        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['pos_body_idx'][i])):
                idx = self._hyperparams['pos_body_idx'][i][j]
                self._model[i]['body_pos'][idx,:] += self._hyperparams['pos_body_offset'][i]
            self._world.set_model(self._model[i])
            init_pose = self._hyperparams['init_pose'][i]
            idx = len(init_pose) // 2
            data = {'qpos': init_pose[:idx], 'qvel': init_pose[idx:]}
            self._world.set_data(data)
            self._world.kinematics()

        # Initialize x0
        self._data = self._world.get_data()
        eepts = self._data['site_xpos'].flatten()

        self._joint_idx = list(range(self._model[0]['nq']))
        self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]

        self.x0 = []
        for x0 in self._hyperparams['init_pose']:
            self.x0.append(np.concatenate([x0, eepts, np.zeros_like(eepts)]))

    def sample(self, policy, T, condition, verbose=True, screenshot_prefix=None):
        """
        Runs a trial and constructs and returns a new sample containing information
        about the trial.

        Args:
            policy: policy to to used in the trial
            T: number of time steps for the trial
            verbose: whether or not to plot the trial
        """
        new_sample = self._init_sample(condition)  # create new sample, populate first time step
        mj_X = self._hyperparams['init_pose'][condition]
        U = np.zeros([T, self.sample_data.dU])
        noise = generate_noise(T, self.sample_data.dU, smooth=self._hyperparams['smooth_noise'], \
                var=self._hyperparams['smooth_noise_var'], \
                renorm=self._hyperparams['smooth_noise_renormalize'])
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx,:] += (var * np.random.randn(1,3))
        self._world.set_model(self._model[condition])
        for t in range(T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t,:], sample=new_sample)
            mj_U = np.clip(mj_U, -self._hyperparams['max_U'], self._hyperparams['max_U'])
            U[t,:] = mj_U
            if verbose:
                self._world.plot(mj_X)
                if screenshot_prefix:
                    imgname = screenshot_prefix+'_'+str(t)+'.png'
                    import pyscreenshot as ImageGrab
                    x = 65
                    y = 50
                    im=ImageGrab.grab(bbox=(x,y,x+640,y+480))
                    im.save(imgname)
            if (t+1) < T:
                for step in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world.step(mj_X, mj_U)
                #TODO: some hidden state stuff will go here
                self._data = self._world.get_data()
                self._set_sample(new_sample, mj_X, t, condition)
        new_sample.set(ACTION, U)
        return new_sample

    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = self.sample_data.create_new()
        sample.set(JOINT_ANGLES, self._hyperparams['init_pose'][condition][self._joint_idx], t=0)
        sample.set(JOINT_VELOCITIES, self._hyperparams['init_pose'][condition][self._vel_idx], t=0)
        self._data = self._world.get_data()
        eepts = self._data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)
        jac = np.zeros([eepts.shape[0], self.x0[condition].shape[0]])
        for site in range(eepts.shape[0] // 3):
            idx = site * 3
            jac[idx:(idx+3), range(self._model[condition]['nq'])] = self._world.get_jac_site(site)
        sample.set(END_EFFECTOR_JACOBIANS, jac, t=0)
        return sample

    def _set_sample(self, sample, mj_X, t, condition):
        sample.set(JOINT_ANGLES, np.array(mj_X[self._joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._vel_idx]), t=t+1)
        curr_eepts = self._data['site_xpos'].flatten()
        sample.set(END_EFFECTOR_POINTS, curr_eepts, t=t+1)
        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (curr_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)
        jac = np.zeros([curr_eepts.shape[0], self.x0[condition].shape[0]])
        for site in range(curr_eepts.shape[0] // 3):
            idx = site * 3
            jac[idx:(idx+3), range(self._model[condition]['nq'])] = self._world.get_jac_site(site)
        sample.set(END_EFFECTOR_JACOBIANS, jac, t=t+1)

    def reset(self, condition):
        pass
