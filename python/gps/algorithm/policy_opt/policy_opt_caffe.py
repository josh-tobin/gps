import caffe
from caffe.proto.caffe_pb2 import SolverParameter
from google.protobuf.text_format import MessageToString
import tempfile

from gps.algorithm.policy_opt.config import policy_opt_caffe


class PolicyOptCaffe(object):
    """Policy optimization using caffe neural network library

    """

    def __init__(self, hyperparams):
        config = copy.deepcopy(policy_opt_caffe)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config)

        self.batch_size = self._hyperparams['batch_size']

        caffe.set_device(self._hyperparams['gpu_id'])
        if self._hyperparams['use_gpu']:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.init_solver()
        # TODO - deal with variance
        # TODO - handle test network assumption a bit nicer, and/or document it
        #self.policy = CaffePolicy(self.solver.test_nets[0], 0) # Is this network immutable?

    def init_solver(self):
        """ Helper method to initialize the solver from hyperparameters. """

        solver_param = SolverParameter()
        solver_param.base_lr = self._hyperparams['lr']
        solver_param.type = self._hyperparams['solver_type']

        # Pass in net parameter either by filename or protostring
        if isinstance(self._hyperparams['network_model'], basestring):
            solver_param.net = self._hyperparams['network_model']  # filename
        else:
            network_arch_params = self._hyperparams['network_arch_params']
            network_arch_params['batch_size'] = self.batch_size
            solver_param.net_param.CopyFrom(self._hyperparams['network_model'](**net_arch_params))

        f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        f.write(MessageToString(solver_param))
        f.close()

        self.solver = caffe.get_solver(f.name)

    # TODO - this assumes that the obs is a vector being passed into the
    # network in the same place (won't work with images or multimodal networks)
    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """ Update policy.

        Args:
            obs: numpy array that is N x Dobs
            tgt_mu: numpy array that is N x Du
            tgt_prc: numpy array that is N x Du x Dx

        Returns:
            a CaffePolicy with updated weights
        """
        # TODO make sure that self.batch_size == solver.net.blobs['DummyDataX'].data.shape[0]?
        # TODO also make sure that obs.shape[0] == tgt_mu.shape[0] == ..., etc?
        # TODO incorporate importance weights tgt_wt
        blob_names = solver.net.blobs.keys()

        batches_per_epoch = np.floor(obs.shape[0] / self.batch_size)

        for itr in range(self._hyperparams['iterations']):
            # Load in data for this batch
            start_idx = itr % batches_per_epoch
            idx = range(start_idx,start_idx+self.batch_size)
            solver.net.blobs[blob_names[0]].data = obs[idx]
            solver.net.blobs[blob_names[1]].data = tgt_mu[idx]
            solver.net.blobs[blob_names[1]].data = tgt_prc[idx]

            self.solver.step(1)

            # To get the training loss:
            #train_loss = solver.net.blobs['loss'].data

            # To run a  test:
            #if itr % test_interval:
            #    print 'Iteration', itr, 'testing...'
            #    solver.test_nets[0].forward()

        # Save out the weights, TODO - figure out how to get itr number
        solver.net.save(self._hyperparams['weights_file_prefix']+'_itr1.caffemodel')

        # TODO - does this do the right thing? Is the policy going to have the new weights?
        return self.policy

    def prob(self, obs):
        # TODO - docs.
        """ Run policy forward.
        """
        # TODO - test net batch size? might need to iterate n/batch_size times
        blob_names = solver.test_nets[0].blobs.keys()
        solver.test_nets[0].blobs[blob_names[0]].data = obs

        return solver.test_nets[0].forward(), []
