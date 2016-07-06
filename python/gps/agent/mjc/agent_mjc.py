""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import mjcpy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MUJOCO
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE, JOINT_ACCELERATIONS

from gps.sample.sample import Sample


class AgentMuJoCo(Agent):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_MUJOCO)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_conditions()
        self._setup_world(hyperparams['filename'])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename',
                      'mass_body_idx', 'mass_body_mult', 
                      'body_color_offset', 
                      'gain_scale', 'ee_points_tgt',
                      'damping_mult', 'friction'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)
    
    def _setup_world(self, filename):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        Todo: refactor this routine
        """
        self._world = []
        self._model = []

        # Initialize Mujoco worlds. If there's only one xml file, create a single world object,
        # otherwise create a different world for each condition.
        if not isinstance(filename, list):
            self._world = mjcpy.MJCWorld(filename)
            self._model = self._world.get_model()
            self._world = [self._world
                           for _ in range(self._hyperparams['conditions'])]
            self._model = [copy.deepcopy(self._model)
                           for _ in range(self._hyperparams['conditions'])]
        else:
            for i in range(self._hyperparams['conditions']):
                self._world.append(mjcpy.MJCWorld(self._hyperparams['filename'][i]))
                self._model.append(self._world[i].get_model())

        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['pos_body_idx'][i])):
                idx = self._hyperparams['pos_body_idx'][i][j]
                # Set body position
                self._model[i]['body_pos'][idx, :] += \
                        self._hyperparams['pos_body_offset'][i]
        # Adjust the mass (and color) of the object
        for i in range(self._hyperparams['conditions']):
            for j in range(len(self._hyperparams['mass_body_idx'][i])):
                idx = self._hyperparams['mass_body_idx'][i][j] 
                
                self._model[i]['body_mass'][idx,:] *= \
                        self._hyperparams['mass_body_mult'][i][j]
                self._model[i]['geom_rgba'][idx,:] += \
                        self._hyperparams['body_color_offset'][i]
            self._world[i].set_model(self._model[i])
            self._reset_world(i)
        for i in range(self._hyperparams['conditions']):
            self._model[i]['dof_damping'] *= \
                    self._hyperparams['damping_mult'][i]
            self._model[i]['dof_frictional'] = np.ones_like(self._model[i]['dof_frictional'])
            self._model[i]['dof_friction'] = self._hyperparams['friction']
        self._joint_idx = list(range(self._model[0]['nq']))
        self._vel_idx = [i + self._model[0]['nq'] for i in self._joint_idx]
        self._included_joint_idx = list(range(self._hyperparams['sensor_dims'][0]))
        print 'included joint index'
        print self._included_joint_idx
        self._included_vel_idx = [i + self._model[0]['nq'] 
                                  for i in self._included_joint_idx]
        print 'included velocity index'
        print self._included_vel_idx
        # Scale the gain of the model
        for i in range(self._hyperparams['conditions']):
            self._model[i]['actuator_gainprm'][:,0] *= self._hyperparams['gain_scale'][i]
        # Initialize x0.
        self.x0 = []
        # Initialize ee_points
        for i in range(self._hyperparams['conditions']):
            # Setup ee_points_tgt. If it's not provided, assume
            # it's zero
            if self._hyperparams['ee_points_tgt'][i] is None:
                self._hyperparams['ee_points_tgt'][i] = \
                        np.zeros_like(self._world[i].get_data()\
                        ['site_xpos'].flatten())
            if END_EFFECTOR_POINTS in self.x_data_types:
                # To be consistent with gazebo, ee points are defined 
                # relative to the shoulder joint
                eepts = self._get_eepts(i)
                adjusted_eepts = self._adjust_eepts(i, eepts)
                self.x0.append(
                    np.concatenate([
                        self._hyperparams['x0'][i][self._included_joint_idx],
                        self._hyperparams['x0'][i][self._included_vel_idx],
                        adjusted_eepts, 
                        np.zeros_like(adjusted_eepts)])
                )
            else:
                self.x0.append(self._hyperparams['x0'][i])
        
        # Setup shoulder position for speed in fwd kinematics: assume it
        # won't change during the trial
        self._shoulder_pos = []
        N_EEPTS = 3
        for i in range(self._hyperparams['conditions']):
            SHOULDER_IDX = 3
            data = self._get_data(i)
            self._shoulder_pos.append(
                    np.concatenate([data['xpos'][SHOULDER_IDX,:].flatten()]*N_EEPTS))
        
        # Setup object ids
        self._object_ids = []
        for i in range(self._hyperparams['conditions']):
            # For now, assume it's the same for all conditions
            self._object_ids.append(self._hyperparams['object_ids'])
        cam_pos = self._hyperparams['camera_pos']
        if 'plot' not in self._hyperparams or self._hyperparams['plot']:
            for i in range(self._hyperparams['conditions']):
                self._world[i].init_viewer(AGENT_MUJOCO['image_width'],
                                       AGENT_MUJOCO['image_height'],
                                       cam_pos[0], cam_pos[1], cam_pos[2],
                                       cam_pos[3], cam_pos[4], cam_pos[5])

    def _set_gravity(self, condition, g=-9.81):
        """ Set the external forces corresponding to the gravity
            on non-robot objects in the scene. This is used for the 
            PR2 simulator to model the spring counterbalances in the PR2
            arm 
        """

        xfrc = self._world[condition].get_data()['xfrc_applied']
        for obj_idx in self._object_ids[condition].values():
            obj_mass = self._model[condition]['body_mass'][obj_idx]
            xfrc[obj_idx, 2] = g * obj_mass # 2 is -z direction
        self._world[condition].set_data({'xfrc_applied': xfrc})
   
    def _reset_world(self, condition):

        x0 = self._hyperparams['x0'][condition]
        idx = int(np.ceil(float(len(x0))/ 2.0))
        data = {'qpos': x0[:idx], 'qvel': x0[idx:]}
        self._world[condition].set_data(data)
        self._world[condition].kinematics()
    

    def sample(self, policy, condition, verbose=True, save=True, screenshot_prefix=None):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
        """
        self._reset_world(condition) 
        # Set external forces corresponding to gravity on objects
        self._set_gravity(condition)
        # Create new sample, populate first time step.
        mb_idx = self._hyperparams['mass_body_idx'][condition]
        new_sample = self._init_sample(condition)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        noise = generate_noise(self.T, self.dU, self._hyperparams)
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                self._model[condition]['body_pos'][idx, :] += \
                        var * np.random.randn(1, 3)
        self._world[condition].set_model(self._model[condition])
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :], sample=new_sample)
            U[t, :] = mj_U
            if verbose:
                self._world[condition].plot(mj_X)
                # obs = self._world[condition].get_image()
                if screenshot_prefix:
                    imgname = screenshot_prefix + '_' + str(t) + '.png'
                    import pyscreenshot as ImageGrab
                    x = 65
                    y = 50
                    im = ImageGrab.grab(bbox=(x, y, x + 640, y + 480))
                    im.save(imgname)
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    mj_X, _ = self._world[condition].step(mj_X, mj_U)
                #TODO: Some hidden state stuff will go here.
                self._data = self._world[condition].get_data()
                self._set_sample(new_sample, mj_X, t, condition)
                if ACTION in self.obs_data_types:
                    new_sample.set(ACTION, mj_U, t=t+1)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample
    
    def step(self, condition, mj_X, mj_U, prev_eepts):
        #mj_X64, mj_U64 = self._change_types(mj_X, mj_U)

        for _ in range(self._hyperparams['substeps']):
            #mj_X, raw_eepts = self._world[condition].step(mj_X.astype(np.float64), 
            #        mj_U.astype(np.float64))
            #mj_X, raw_eepts = self._just_step(condition, mj_X64, mj_U64)
            mj_X, raw_eepts = self._just_step(condition, mj_X, mj_U)
        eepts = self._modify_eepts(condition, raw_eepts)
        eept_vels = (eepts - prev_eepts) / self._hyperparams['dt']
        return np.concatenate([mj_X, eepts, eept_vels])
    
    def _change_types(self, mj_X, mj_U):
        return mj_X.astype(np.float64), mj_U.astype(np.float64)
    # For debugging
    def _just_step(self, condition, mj_X, mj_U):
        return self._world[condition].step(mj_X, 
                                           mj_U)

    def _calc_relative_to_shoulder(self, condition, eepts):
        n_eepts = eepts.shape[0]
        #eepts_relative_to_shoulder = (eepts.flatten() - 
        #        np.concatenate([self._shoulder_pos[condition]]*n_eepts))
        eepts_relative_to_shoulder = eepts.flatten() - self._shoulder_pos[condition]
        return eepts_relative_to_shoulder
    def _calc_relative_to_target(self, condition, relative_to_shoulder):
        eepts_relative_to_target = (relative_to_shoulder -
                self._hyperparams['ee_points_tgt'][condition].flatten())
        return eepts_relative_to_target
    def _modify_eepts(self, condition, eepts):
        ''' To-do: we have a lot of fns like this: need to combine '''
        n_eepts = eepts.shape[0]
        eepts_relative_to_shoulder = self._calc_relative_to_shoulder(condition, eepts)
        eepts_relative_to_target = self._calc_relative_to_target(condition, eepts_relative_to_shoulder)
        return eepts_relative_to_target

    @property
    def worlds(self):
        return self._world

    def _get_eepts(self, condition):

        SHOULDER_IDX = 3
        data = self._get_data(condition)
        shoulder_pos = data['xpos'][SHOULDER_IDX,:].flatten()
        abs_eepts  = data['site_xpos']
        n_eepts = abs_eepts.shape[0]
        return abs_eepts.flatten() - np.concatenate([shoulder_pos]*n_eepts)
    
    def _combine_eepts(self, data):
        shoulder_pos = data['xpos'][SHOULDER_IDX,:].flatten()
        abs_eepts = data['site_xpos']
        n_eepts = abs_eepts.shape[0]
        return abs_eepts.flatten() - np.concatenate([shoulder_pos]*n_eepts)

    def _get_data(self, condition):
        data = self._world[condition].get_data()
        return data

    def _adjust_eepts(self, condition, eepts):
        if self._hyperparams['ee_point_mode'] == 'stationary_target':        
            adjusted_eepts = eepts - \
                    self._hyperparams['ee_points_tgt'][condition].flatten()
        elif self._hyperparams['ee_point_mode'] == 'moving_target':
            eept_dim = 3 # dimensionality of a single eept
            ee_indices = self._hyperparams['ee_points_indices']
            ee_tgt = self._hyperparams['ee_points_tgt_site']
            adjusted_eepts = eepts[eept_dim*ee_indices:
                                   eept_dim*(ee_indices+1)].flatten() \
                             - eepts[eept_dim*ee_tgt:
                                     eept_dim*(ee_tgt+1)].flatten()
        else:
            raise Exception('Unsupported ee_point_mode %s.'%self._hyperparams['ee_point_mode'])
        return adjusted_eepts
    
    def _get_ee_point_jacobians(self, condition, adjusted_eepts):
        nJ = len(self._included_joint_idx)
        jac = np.zeros([adjusted_eepts.shape[0], nJ])
        for site in range(adjusted_eepts.shape[0] // 3):
            idx = site * 3
            jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site) \
                                    [:, self._included_joint_idx]
        return jac

    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)
        sample.set(JOINT_ANGLES,
                   self._hyperparams['x0'][condition][self._included_joint_idx], t=0)
        sample.set(JOINT_VELOCITIES,
                   self._hyperparams['x0'][condition][self._included_vel_idx], t=0)
        self._data = self._world[condition].get_data()
        
        if JOINT_ACCELERATIONS in self.x_data_types:
            sample.set(JOINT_ACCELERATIONS,
                       np.zeros_like(self._hyperparams['x0'][condition]\
                                     [self._included_vel_idx]), t=0)
        if END_EFFECTOR_POINTS in self.x_data_types:
            # To be consistent with gazebo, ee points are defined 
            # relative to the shoulder joint

            eepts = self._get_eepts(condition)
            adjusted_eepts = self._adjust_eepts(condition, eepts)

            sample.set(END_EFFECTOR_POINTS, adjusted_eepts, t=0)
            sample.set(END_EFFECTOR_POINT_VELOCITIES, 
                        np.zeros_like(adjusted_eepts), t=0)
            #jac = np.zeros([adjusted_eepts.shape[0], self._model[condition]['nv']])

            jac = self._get_ee_point_jacobians(condition, adjusted_eepts)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)

        # save initial image to meta data
        self._world[condition].plot(self._hyperparams['x0'][condition])
        img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                      self._hyperparams['image_height'])
        if ACTION in self.obs_data_types:
            sample.set(ACTION, np.zeros(7), t=0)
        # mjcpy image shape is [height, width, channels],
        # dim-shuffle it for later conv-net processing,
        # and flatten for storage
        img_data = np.transpose(img["img"], (2, 1, 0)).flatten()
        # if initial image is an observation, replicate it for each time step
        if CONTEXT_IMAGE in self.obs_data_types:
            sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
        else:
            sample.set(CONTEXT_IMAGE, img_data, t=None)
        sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
                                                self._hyperparams['image_width'],
                                                self._hyperparams['image_height']]), t=None)
        # only save subsequent images if image is part of observation
        if RGB_IMAGE in self.obs_data_types:
            sample.set(RGB_IMAGE, img_data, t=0)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
        return sample
    
    def get_ee_obs(self, prev_eepts, condition):
        curr_eepts = self._get_eepts(condition)
        curr_adj_eepts = self._adjust_eepts(condition, curr_eepts)

        eept_vels = (curr_adj_eepts - prev_eepts) / self._hyperparams['dt']
        jac = self._get_ee_point_jacobians(condition, curr_adj_eepts)
        return curr_adj_eepts, eept_vels, jac
    
    def _set_sample(self, sample, mj_X, t, condition):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        
        sample.set(JOINT_ANGLES, np.array(mj_X[self._included_joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._included_vel_idx]), t=t+1)
        
        
        if JOINT_ACCELERATIONS in self.x_data_types:
            curr_velocities = sample.get(JOINT_VELOCITIES, t=t+1)
            prev_velocities = sample.get(JOINT_VELOCITIES, t=t)
            joint_accelerations = (curr_velocities - prev_velocities) \
                                  / self._hyperparams['dt']
            sample.set(JOINT_ACCELERATIONS, joint_accelerations, t=t+1)
        if END_EFFECTOR_POINTS in self.x_data_types:
            #curr_eepts = self._get_eepts(condition)
            #curr_adj_eepts = self._adjust_eepts(condition, curr_eepts)
            prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
            curr_adj_eepts, eept_vels, jac = \
                    self.get_ee_obs(prev_eepts, condition)
            sample.set(END_EFFECTOR_POINTS, curr_adj_eepts, t=t+1)
            #eept_vels = (curr_adj_eepts - prev_eepts) / self._hyperparams['dt']
            sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)
            #jac = np.zeros([curr_adj_eepts.shape[0], 
            #                self._model[condition]['nv']])
            #for site in range(curr_adj_eepts.shape[0] // 3):
            #    idx = site * 3
            #    jac[idx:(idx+3), :] = self._world[condition].get_jac_site(site)
            #jac = self._get_ee_point_jacobians(condition, curr_adj_eepts)
            sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t+1)
        
        if RGB_IMAGE in self.obs_data_types:
            img = self._world[condition].get_image_scaled(self._hyperparams['image_width'],
                                                          self._hyperparams['image_height'])
            sample.set(RGB_IMAGE, np.transpose(img["img"], (2, 1, 0)).flatten(), t=t+1)

    def _get_image_from_obs(self, obs):
        imstart = 0
        imend = 0
        image_channels = self._hyperparams['image_channels']
        image_width = self._hyperparams['image_width']
        image_height = self._hyperparams['image_height']
        for sensor in self._hyperparams['obs_include']:
            # Assumes only one of RGB_IMAGE or CONTEXT_IMAGE is present
            if sensor == RGB_IMAGE or sensor == CONTEXT_IMAGE:
                imend = imstart + self._hyperparams['sensor_dims'][sensor]
                break
            else:
                imstart += self._hyperparams['sensor_dims'][sensor]
        img = obs[imstart:imend]
        img = img.reshape((image_channels, image_width, image_height))
        img = np.transpose(img, [1, 2, 0])
        return img
