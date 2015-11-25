import caffe
from caffe.proto.caffe_pb2 import SolverParameter, TRAIN, TEST
import copy
from google.protobuf.text_format import MessageToString
import logging
import numpy as np
import tempfile

from gps.algorithm.policy.caffe_policy import CaffePolicy
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy_opt.config import policy_opt_caffe


LOGGER = logging.getLogger(__name__)

class PolicyOptCaffe(PolicyOpt):
    """Policy optimization using caffe neural network library

    """

    def __init__(self, hyperparams, dObs, dU):
        config = copy.deepcopy(policy_opt_caffe)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config, dObs, dU)

        self.batch_size = self._hyperparams['batch_size']

        caffe.set_device(self._hyperparams['gpu_id'])
        if self._hyperparams['use_gpu']:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.init_solver()
        self.var = self._hyperparams['init_var'] * np.ones(dU)

        self.policy = CaffePolicy(self.solver.test_nets[0], np.zeros(dU))

    def init_solver(self):
        """ Helper method to initialize the solver from hyperparameters. """

        solver_param = SolverParameter()
        solver_param.display = 0  # Don't display anything.
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.lr_policy = self._hyperparams['lr_policy']
        solver_param.momentum = self._hyperparams['momentum']
        solver_param.weight_decay = self._hyperparams['weight_decay']
        solver_param.type = self._hyperparams['solver_type']

        # Pass in net parameter either by filename or protostring
        if isinstance(self._hyperparams['network_model'], basestring):
            self.solver = caffe.get_solver(self._hyperparams['network_model'])
        else:
            network_arch_params = self._hyperparams['network_arch_params']
            network_arch_params['dim_input'] = self._dObs
            network_arch_params['dim_output'] = self._dU

            network_arch_params['batch_size'] = self.batch_size
            network_arch_params['phase'] = TRAIN
            solver_param.train_net_param.CopyFrom(self._hyperparams['network_model'](**network_arch_params))

            network_arch_params['batch_size'] = 1
            network_arch_params['phase'] = TEST
            solver_param.test_net_param.add().CopyFrom(self._hyperparams['network_model'](**network_arch_params))

            # These are required by caffe to be set, but not used.
            solver_param.test_iter.append(1)
            solver_param.test_interval = 1000000

            f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            f.write(MessageToString(solver_param))
            f.close()

            self.solver = caffe.get_solver(f.name)

    # TODO - this assumes that the obs is a vector being passed into the
    # network in the same place (won't work with images or multimodal networks)
    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """ Update policy.

        Args:
            obs: numpy array that is N x dObs
            tgt_mu: numpy array that is N x dU
            tgt_prc: numpy array that is N x dU x dU
            tgt_wt: numpy array that is N x T

        Returns:
            a CaffePolicy with updated weights
        """
        # TODO also make sure that obs.shape[0] == tgt_mu.shape[0] == ..., etc?
        # TODO - normalization?
        N, T = obs.shape[:2]
        dU, dObs = self._dU, self._dObs
        obs = np.reshape(obs, (N*T, dObs))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))
        tgt_prc = tgt_wt * tgt_prc

        blob_names = self.solver.net.blobs.keys()

        # Assuming that N*T >= self.batch_size
        batches_per_epoch = np.floor(N*T / self.batch_size)
        idx = range(N*T)
        np.random.shuffle(idx)
        for itr in range(self._hyperparams['iterations']):
            # Load in data for this batch
            start_idx = int(itr % batches_per_epoch)
            idx_i = idx[start_idx:start_idx+self.batch_size]
            self.solver.net.blobs[blob_names[0]].data[:] = obs[idx_i]
            self.solver.net.blobs[blob_names[1]].data[:] = tgt_mu[idx_i]
            self.solver.net.blobs[blob_names[2]].data[:] = tgt_prc[idx_i]

            self.solver.step(1)

            # To get the training loss:
            train_loss = self.solver.net.blobs[blob_names[-1]].data
            if itr % 1000 == 0:
                LOGGER.debug('Caffe iteration %d, loss %f', itr, train_loss)

            # To run a  test
            #if itr % test_interval:
            #    print 'Iteration', itr, 'testing...'
            #    solver.test_nets[0].forward()

        # Save out the weights, TODO - figure out how to get itr number
        self.solver.net.save(self._hyperparams['weights_file_prefix']+'_itr1.caffemodel')

        # Optimize variance
        A = np.sum(tgt_prc,0) + 2*N*T*self._hyperparams['ent_reg']*np.ones((dU,dU))
        A = A / np.sum(tgt_wt)

        # TODO - use dense covariance?
        self.var = 1 / np.diag(A)

        self.policy.net.share_with(self.solver.net)
        return self.policy

    def prob(self, obs):
        """ Run policy forward.

        Args:
            obs: numpy array that is N x T x Dobs

        Returns:
            pol mu, pol var, pol prec, pol det sigma
        """
        dU = self._dU
        N, T = obs.shape[:2]

        output = np.zeros([N, T, dU])
        blob_names = self.solver.test_nets[0].blobs.keys()

        self.solver.test_nets[0].share_with(self.solver.net)

        for i in range(N):
            for t in range(T):
                # Feed in data
                self.solver.test_nets[0].blobs[blob_names[0]].data[:] = obs[i,t]

                # Assume that the first output blob is what we want
                output[i,t,:] = self.solver.test_nets[0].forward().values()[0][0]

        pol_sigma = np.tile(np.diag(self.var), (N, T, 1, 1))
        pol_prec = np.tile(np.diag(1 / self.var), (N, T, 1, 1))
        pol_det_sigma = np.tile(np.prod(self.var), (N, T))

        return output, pol_sigma, pol_prec, pol_det_sigma
