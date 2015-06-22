import abc


class TrajOpt(object):
    """Trajectory optimization superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data, dynamics):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.dynamics = dynamics

    @abc.abstractmethod
    def update(self):
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass")

# TODO - need interface with C++ trajopt
