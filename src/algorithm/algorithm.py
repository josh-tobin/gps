#!/usr/bin/env python

class Algorithm():
    """Algorithm superclass

    """
    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.cost = Cost(hyperparams['cost'], sample_data)
        self.dynamics = Dynamics(hyperparams['dynamics'], sample_data)
        self.policy_opt = None
        self.traj_opt = None

    def iteration(self):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass");

    def update_vars(self):
        """ Update variables the algorithm. """
        raise NotImplementedError("Must be implemented in subclass");


