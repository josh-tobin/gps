import abc


class PolicyOpt(object):
    """Policy optimization superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def update(self):
        """ Update policy. """
        raise NotImplementedError("Must be implemented in subclass");
