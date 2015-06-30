import abc
import numpy as np

class Dynamics(object):
    """Dynamics superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self._sample_data = sample_data

        # TODO - Currently assuming that dynamics will always be linear with X.

        # TODO - allocate arrays using hyperparams dU, dX, T
        # Fitted dynamics: x_t+1 = Fm * [x_t;u_t] + fv
        self.Fm = np.array(np.nan)
        self.fv = np.array(np.nan)
        self.dyn_covar = np.array(np.nan)  # Covariance

    @abc.abstractmethod
    def update_prior(self):
        """ Update dynamics prior. """
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def fit(self):
        """ Fit dynamics. """
        raise NotImplementedError("Must be implemented in subclass")
