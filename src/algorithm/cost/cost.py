import abc


class Cost(object):
    """Cost superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self._sample_data = sample_data

    @abc.abstractmethod
    def eval(self, x, u, obs, sample_meta):
        """
        Evaluate cost function and derivatives

        Args:
            x: A T x Dx state matrix
            u: A T x Du action matrix
            obs: A T x Dobs observation matrix
            sample_meta: List of cost_info objects
                (temporary placeholder until we discuss how to pass these around)
        Return:
            l, lx, lu, lxx, luu, lux: Loss (Tx1 float) and 1st/2nd derivatives.
        """
        raise NotImplementedError("Must be implemented in subclass")
