''' Defines tests for keras model input to tensorflow policy opt '''
import os
import os.path
import sys
import numpy as np
#import tensorflow as tf

# Add gps/python to path
gps_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'python'))
sys.path.append(gps_path)

from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy_opt.config import POLICY_OPT_KERAS, \
                                            POLICY_OPT_KERAS_RNN

def test_keras_policy_init():
    hyperparams = POLICY_OPT_KERAS
    deg_obs = 100
    deg_action = 7
    PolicyOptTf(hyperparams, deg_obs, deg_action)

def test_keras_rnn_init():
    hyperparams = POLICY_OPT_KERAS_RNN
    deg_obs = 20
    deg_action = 7
    PolicyOptTf(hyperparams, deg_obs, deg_action)

def test_keras_forward():
    hyperparams = POLICY_OPT_KERAS
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyperparams, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale)
                                                       , axis=0)
    policy_opt.prob(obs=obs)

def test_keras_forward_noise():
    hyperparams = POLICY_OPT_KERAS
    deg_obs = 100
    deg_action = 7
    policy_opt = PolicyOptTf(hyperparams, deg_obs, deg_action)
    N = 20
    T = 30
    obs = np.random.randn(N, T, deg_obs)
    obs_reshaped = np.reshape(obs, (N*T, deg_obs))
    policy_opt.policy.scale = np.diag(1.0 / np.std(obs_reshaped, axis=0))
    policy_opt.policy.bias = -np.mean(obs_reshaped.dot(policy_opt.policy.scale)
                                                       , axis=0)
    noise = np.random.randn(deg_action)
    policy_opt.policy.act(None, obs[0,0], None, noise)

def test_policy_opt_backwards():
    hyperparams = POLICY_OPT_KERAS
    deg_obs = 20
    deg_action = 7
    policy_opt = PolicyOptTf(hyperparams, deg_obs, deg_action)
    policy_opt._hyperparams['iterations'] = 100
    N = 10
    T = 10
    obs = np.random.randn(N, T, deg_obs)
    tgt_mu = np.random.randn(N, T, deg_action)
    tgt_prc = np.random.randn(N, T, deg_action, deg_action)
    tgt_wt = np.random.randn(N, T)
    new_policy = policy_opt.update(obs, tgt_mu, tgt_prc, tgt_wt, itr=0, 
                                   inner_itr=1)

def test_policy_opt_live():
    test_dir = os.path.dirname(os.path.abspath(__file__)) + '/test_data/'
    obs = np.load(test_dir + 'obs.npy')
    tgt_mu = np.load(test_dir + 'tgt_mu.npy')
    tgt_prc = np.load(test_dir + 'tgt_prc.npy')
    scale = np.load(test_dir + 'scale_npy.npy')
    bias = np.load(test_dir + 'bias_npy.npy')
    hyperparams = POLICY_OPT_KERAS
    deg_obs = 4
    deg_action = 2

    policy = PolicyOptTf(hyperparams, deg_obs, deg_action)
    policy.policy.scale = scale
    policy.policy.bias = bias

    iterations = 200
    batch_size = 32
    batches_per_epoch = np.floor(800 / batch_size)
    idx = range(800)
    np.random.shuffle(idx)

    for i in range(iterations):
        # load in data for this batch
        start_idx = int(i * batch_size %
                        (batches_per_epoch * batch_size))
        idx_i = idx[start_idx:start_idx + batch_size]
        feed_dict = {policy.obs_tensor: obs[idx_i],
                     policy.action_tensor: tgt_mu[idx_i],
                     policy.precision_tensor: tgt_prc[idx_i],
                     }
        t = policy.sess.run(policy.act_op, 
                feed_dict={policy.obs_tensor: np.expand_dims(obs[idx_i][0], 0)})
        policy.solver(feed_dict, policy.sess)
    
def main():
    print('running keras policy tests')
    test_keras_policy_init()
    test_keras_rnn_init()
    test_keras_forward()
    test_keras_forward_noise()
    test_policy_opt_backwards()
    test_policy_opt_live()
    print('keras policy tests passed')

if __name__ == '__main__':
    main()
