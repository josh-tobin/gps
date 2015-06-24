from copy import deepcopy

from algorithm import Algorithm
from config import alg_traj_opt

class AlgorithmTrajOpt(Algorithm):
    """Sample-based trajectory optimization.

    """
    def __init__(self, hyperparams, sample_data):
        # TODO - Initialize trajectory distributions somewhere
        config = deepcopy(alg_traj_opt)
        config.update(hyperparams)
        Algorithm.__init__(self, config, sample_data)

        self.traj_opt = hyperparams['traj_opt']['type'](hyperparams['traj_opt'], sample_data, self.dynamics)

    def iteration(self):
        """ Run iteration of the LQR. """

        # Update dynamics model using all sample.
        self.dynamics.update_prior()
        self.dynamics.fit()

        self.update_cost()
        self.update_vars()
        for inner_itr in range(self._hyperparams['inner_iterations']):
            self.traj_opt.update()

    def update_cost(self):
        raise NotImplementedError("TODO")

    def update_vars(self):
        raise NotImplementedError("TODO")
