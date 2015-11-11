import abc


class PolicyOpt(object):
    """Policy optimization superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, dObs, dU):
        self._hyperparams = hyperparams
        self._dObs = dObs
        self._dU = dU

    @abc.abstractmethod
    def update(self):
        """ Update policy. """
        raise NotImplementedError("Must be implemented in subclass");
