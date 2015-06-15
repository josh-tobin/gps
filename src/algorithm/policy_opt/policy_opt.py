import abc


class PolicyOpt(object):
    """Policy optimization superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    @abc.abstractmethod
    def update(self):
        """ Update cost policy. """
        raise NotImplementedError("Must be implemented in subclass");
