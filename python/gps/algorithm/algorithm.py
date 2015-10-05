import abc


class Algorithm(object):
    """Algorithm superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.traj_opt = hyperparams['traj_opt']['type'](hyperparams['traj_opt'])
        self.cost = [hyperparams['cost']['type'](hyperparams['cost'])]*hyperparams['conditions']

    @abc.abstractmethod
    def iteration(self, sample_data):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")
