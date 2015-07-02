import abc


class Cost(object):
    """Cost superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    @abc.abstractmethod
    # TODO - Utilize self.sample_data rather than pass in a sample
    def eval(self, sample):
        """
        Evaluate cost function and derivatives

        Args:
            sample: A Sample object

        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        raise NotImplementedError("Must be implemented in subclass")
