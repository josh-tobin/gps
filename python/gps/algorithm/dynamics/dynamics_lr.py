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
        self.dyn_covar = np.zeros([T, dX, dX])

        # Fit dynamics wih least squares regression
        for t in range(T-1):
            result, _, _, _ = np.linalg.lstsq(np.c_[X[:, t, :], U[:, t, :], np.ones(N)], np.c_[X[:, t+1, :], np.ones(N)])
            Fm = result[:-1, :-1].T
            self.Fm[t, :, :] = Fm
            self.fv[t, :] = result[-1,:-1]

            x_next_covar = np.cov(X[:, t+1, :].T)
            xu_covar = np.cov(np.c_[X[:, t, :], U[:, t, :]].T)
            dyn_covar = x_next_covar - Fm.dot(xu_covar).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5*(dyn_covar+dyn_covar.T)  # Make symmetric
