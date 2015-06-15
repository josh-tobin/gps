import abc


class Dynamics(object):
    """Dynamics superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    @abc.abstractmethod
    def update(self):
        """ Update dynamics. """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def eval(self):
        """ Evaluate dynamics. """
        raise NotImplementedError("Must be implemented in subclass")
