import abc


class Algorithm(object):
    """Algorithm superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.cost = Cost(hyperparams['cost'], sample_data)
        self.dynamics = Dynamics(hyperparams['dynamics'], sample_data)
        self.policy_opt = None
        self.traj_opt = None

    @abc.abstractmethod
    def iteration(self):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def update_vars(self):
        """ Update variables the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")


