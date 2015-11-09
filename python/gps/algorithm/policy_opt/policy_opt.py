import abc


class PolicyOpt(object):
    """Policy optimization superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, dObs):
        self._hyperparams = hyperparams
        self._dObs = dObs

    @abc.abstractmethod
    def update(self):
        """ Update policy. """
        raise NotImplementedError("Must be implemented in subclass");
