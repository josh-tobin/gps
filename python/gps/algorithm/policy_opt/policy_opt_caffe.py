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

        SolverParameter solver_param = new SolverParameter()
        solver_param.base_lr = self._hyperparams['lr']

        # Pass in net parameter either by filename or protostring
        if isinstance(self._hyperparams['network_model'], basestring):
            solver_param.train_net = self._hyperparams['network_model']
        else:
            solver_param.train_net_param = self._hyperparams['network_model'](**self._hyperparams['network_params'])

        self.solver = config['solver'](solver_param)

    def update(self):
        """ Update policy.

        Returns:
            a CaffePolicy with updated weights
        """

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

