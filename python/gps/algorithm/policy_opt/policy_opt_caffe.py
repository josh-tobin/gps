from config import policy_opt_caffe
import caffe
from caffe.proto.caffe_pb2 import SolverParameter


class PolicyOptCaffe(object):
    """Policy optimization using caffe neural network library

    """

    def __init__(self, hyperparams):
        config = copy.deepcopy(policy_opt_caffe)
        config.update(hyperparams)

        PolicyOpt.__init__(self, config)

        caffe.set_device(self._hyperparams['gpu_id'])
        if self._hyperparams['use_gpu']:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.init_solver()
        # TODO - deal with variance
        # TODO - handle test network assumption a bit nicer, and/or document it
        self.policy = CaffePolicy(self.solver.test_nets[0], 0) # Is this network immutable?

    def init_solver(self):
        """ Helper method to initialize the solver from hyperparameters. """

        solver_param = SolverParameter()
        solver_param.base_lr = self._hyperparams['lr']
        #solver_param.type = self._hyperparams['solver_type']

        # Pass in net parameter either by filename or protostring
        if isinstance(self._hyperparams['network_model'], basestring):
            solver_param.net = self._hyperparams['network_model']  # filename
        else:
            solver_param.net_param.CopyFrom(self._hyperparams['network_model'](**self._hyperparams['network_params']))

        self.solver = config['solver'](solver_param) # TODO - this won't work yet... :/

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        """ Update policy.

        Returns:
            a CaffePolicy with updated weights
        """
        # TODO - this next line won't work yet. Also, need to convert to single?
        solver.net.SetInputArrays(obs,tgt_mu,tgt_prc,tgt_wt)

        for itr in range(self._hyperparams['iterations']):
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
        # TODO - float?
        solver.test_nets[0].SetInputArrays(obs)

        return solver.test_nets[0].forward(), []
