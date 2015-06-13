import abc

class Cost(object):
    """Cost superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    @abc.abstractmethod
    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError("Must be implemented in subclass");
