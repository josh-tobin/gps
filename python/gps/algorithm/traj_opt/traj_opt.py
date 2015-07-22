import abc


class TrajOpt(object):
    """Trajectory optimization superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def update(self):
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass")

# TODO - need interface with C++ trajopt
