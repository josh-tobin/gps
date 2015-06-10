#!/usr/bin/env python

class Algorithm():
    """Algorithm superclass

    """
    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.cost = Cost(hyperparams['cost'])
        self.dynamics = Dynamics(hyperparams['dynamics'])
        self.policy_opt = None
        self.traj_opt = None

    def iteration(self):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass");

    def update_vars(self):
        """ Update variables the algorithm. """
        raise NotImplementedError("Must be implemented in subclass");


class BasicLQR(Algorithm):
    """Trajectory Optimization

    """
    def __init__(self, hyperparams, sample_data):
        # TODO - need to add a helper to initialize traj distributions.
        super(BasicLQR,self).__init__(hyperparams,sample_data)
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
