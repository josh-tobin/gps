import abc


class Cost(object):
    """Cost superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    def eval(self, sample_idx):
        """
        Evaluate cost function and derivatives

        Args:
            sample_idx:  A single index into sample_data

        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        sample = self.sample_data.get_samples(idx=[sample_idx])[0]
        return self.eval_sample(sample)

    @abc.abstractmethod
    def eval_sample(self, sample):
        """
        Evaluate cost function and derivatives

        Args:
            sample:  A sample object

        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        raise NotImplementedError("Must be implemented in subclass")