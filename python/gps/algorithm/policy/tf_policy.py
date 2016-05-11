import pickle

import numpy as np
import tensorflow as tf

from gps.algorithm.policy.policy import Policy


class TfPolicy(Policy):
    """
    A neural network policy implemented in tensor flow. The network output is
    taken to be the mean, and Gaussian noise is added on top of it.
    U = net.forward(obs) + noise, where noise ~ N(0, diag(var))
    Args:
        obs_tensor: tensor representing tf observation. Used in feed dict for forward pass.
        act_op: tf op to execute the forward pass. Use sess.run on this op.
        var: Du-dimensional noise variance vector.
        sess: tf session.
        device_string: tf device string for running on either gpu or cpu.
    """
    def __init__(self, dU, obs_tensor, act_op, var, sess, device_string,
                 hidden_state_tensor=None, initial_hidden_state_tensor=None):
        Policy.__init__(self)
        self.dU = dU
        self.obs_tensor = obs_tensor
        self.act_op = act_op
        self.sess = sess
        self.device_string = device_string
        self.chol_pol_covar = np.diag(np.sqrt(var))
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.hidden_state_tensor = hidden_state_tensor
        self.initial_hidden_state_tensor = initial_hidden_state_tensor
        self.recurrent = False
        if self.hidden_state_tensor is not None:
            self.recurrent = True
            self.hidden_dim = self.initial_hidden_state_tensor.get_shape().\
                                                               as_list()[-1]
            self.hidden_state = np.zeros([1, self.hidden_dim])
    
    
    def act(self, x, obs, t, noise):
        """
        Return an action for a state.
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: Action noise. This will be scaled by the variance.
        """
        # Normalize obs.
        obs = obs.dot(self.scale) + self.bias

        if self.recurrent:
            fd = {self.obs_tensor: np.expand_dims(np.expand_dims(obs, 0), 0),
                         self.initial_hidden_state_tensor: self.hidden_state}
            with tf.device(self.device_string):
                action_mean = self.sess.run(self.act_op, feed_dict=fd)
                self.hidden_state = self.sess.run(self.hidden_state_tensor, 
                                           feed_dict=fd)
        else:
            feed_dict = {self.obs_tensor: np.expand_dims(obs, 0)}
            with tf.device(self.device_string):
                action_mean = self.sess.run(self.act_op, feed_dict=feed_dict)
        if noise is None:
            u = action_mean
        else:
            u = action_mean + self.chol_pol_covar.T.dot(noise)
        if self.recurrent:
            u_out = u[0,0,:]
        else:
            u_out = u[0]
        return u_out  # this algorithm is batched by default. But here, we run with a batch size of one.

    def reset(self):
        if self.recurrent:
            print "RESETTING"
            self.hidden_state = np.zeros([1, self.hidden_dim])

    def pickle_policy(self, deg_obs, deg_action, checkpoint_path):
        """
        We can save just the policy if we are only interested in running forward at a later point
        without needing a policy optimization class. Useful for debugging and deploying.
        """
        pickled_pol = {'deg_obs': deg_obs, 'deg_action': deg_action, 
                       'chol_pol_covar': self.chol_pol_covar,
                       'checkpoint_path_tf': checkpoint_path + '_tf_data', 
                       'scale': self.scale, 'bias': self.bias,
                       'device_string': self.device_string,
                       'goal_state': goal_state, 'x_idx': self.x_idx}
        pickle.dump(pickled_pol, open(checkpoint_path, "wb"))
        saver = tf.train.Saver()
        saver.save(self.sess, checkpoint_path + '_tf_data')

    @classmethod
    def load_policy(cls, policy_dict_path, tf_generator, network_config=None):
        """
        For when we only need to load a policy for the forward pass. For instance, to run on the robot from
        a checkpointed policy.
        """
        from tensorflow.python.framework import ops
        ops.reset_default_graph()  # we need to destroy the default graph before re_init or checkpoint won't restore.
        pol_dict = pickle.load(open(policy_dict_path, "rb"))
        tf_map = tf_generator(dim_input=pol_dict['deg_obs'], 
                              dim_output=pol_dict['deg_action'], batch_size=1,
                              network_config=network_config)

        sess = tf.Session()
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        saver = tf.train.Saver()
        check_file = pol_dict['checkpoint_path_tf']
        saver.restore(sess, check_file)

        device_string = pol_dict['device_string']

        cls_init = cls(pol_dict['deg_action'], tf_map.get_input_tensor(), tf_map.get_output_op(), np.zeros((1,)),
                       sess, device_string)
        cls_init.chol_pol_covar = pol_dict['chol_pol_covar']
        cls_init.scale = pol_dict['scale']
        cls_init.bias = pol_dict['bias']
        cls_init.x_idx = pol_dict['x_idx']
        return cls_init

