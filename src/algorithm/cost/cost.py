import abc


class Cost(object):
    """Cost superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self._sample_data = sample_data

    @abc.abstractmethod
    def eval(self, sample):
        """
        Evaluate cost function and derivatives

        Args:
            sample: A Sample object

        Return:
            l, lx, lu, lxx, luu, lux: Loss (Tx1 float) and 1st/2nd derivatives.
        """
        raise NotImplementedError("Must be implemented in subclass")
