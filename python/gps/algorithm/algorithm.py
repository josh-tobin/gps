import abc


class Algorithm(object):
    """Algorithm superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self._sample_data = sample_data
        self.traj_opt = hyperparams['traj_opt']
        self.cost = hyperparams['cost']

    @abc.abstractmethod
    def iteration(self):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")
