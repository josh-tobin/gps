#!/usr/bin/env python

class LQRAlgorithm(Algorithm):
    """Sample-based trajectory optimization with LQR

    """
    def __init__(self, hyperparams, sample_data):
        # TODO - need to add a helper to initialize traj distributions.
        super(LQRAlgorithm,self).__init__(hyperparams,sample_data)
        self.traj_opt = TrajOptLQR(hyperparams['traj_opt'])

    def iteration(self):
        """ Run iteration of the LQR. """
        self.dynamics.update()
        self.cost.update()
        self.update_vars()
        for inner_itr in range(self._hyperparams['inner_iterations']):
            self.traj_opt.update()

    def update_vars(self):
        raise NotImplementedError("TODO");
