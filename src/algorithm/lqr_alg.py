from algorithm import Algorithm

class LQRAlgorithm(Algorithm):
    """Sample-based trajectory optimization with LQR

    """
    def __init__(self, hyperparams, sample_data):
        # TODO - need to add a helper to initialize traj distributions.
        Algorithm.__init__(self, hyperparams, sample_data)
        self.traj_opt = hyperparams['traj_opt']['type'](hyperparams['traj_opt'], sample_data)

    def iteration(self):
        """ Run iteration of the LQR. """
        self.dynamics.update()
        self.update_cost()
        self.update_vars()
        for inner_itr in range(self._hyperparams['inner_iterations']):
            self.traj_opt.update()

    def update_cost(self):
        raise NotImplementedError("TODO")

    def update_vars(self):
        raise NotImplementedError("TODO")
