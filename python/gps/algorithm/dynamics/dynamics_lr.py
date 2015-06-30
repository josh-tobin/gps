import numpy as np

from dynamics import Dynamics


class DynamicsLR(Dynamics):
    """Dynamics with linear regression, with constant prior.

    """
    def __init__(self,hyperparams, sample_data):
        Dynamics.__init__(self, hyperparams, sample_data)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None

    def update_prior(self):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def fit(self):
        """ Fit dynamics. """
        X = self._sample_data.get_X()  # Use all samples to fit dynamics.
        U = self._sample_data.get_U()
        N, T, dX = X.shape
        dU = U.shape[2]

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])

        # Fit dynamics wih least squares regression
        for t in range(T-1):
            result, _, _, _ = np.linalg.lstsq(np.c_[X[:,t,:],U[:,t,:],np.ones(N)], np.c_[X[:,t+1,:],np.ones(N)])
            self.Fm[t,:,:] = result[:-1,:-1].T
            self.fv[t,:] = result[-1,:-1]
        # TODO - leave last time step as zeros?
        # TODO - what to do with covariance? (the old dynsig)
        self.dyn_covar = np.tile(np.eye(dX), [T, 1, 1])
