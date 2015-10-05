import numpy as np

from dynamics import Dynamics
from dynamics_prior_gmm import DynamicsPriorGMM


class DynamicsLRPrior(Dynamics):
    """Dynamics with linear regression, with arbitrary prior.

    """
    def __init__(self,hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None

        #TODO: Use hyperparams
        self.prior = DynamicsPriorGMM()

    def update_prior(self, sample):
        """ Update dynamics prior. """
        X = sample.get_X()
        U = sample.get_U()
        self.prior.update(X, U)

    def get_prior(self):
        return self.prior

    #TODO: Merge this with DynamicsLR.fit - lots of duplicated code
    def fit(self, sample_data):
        """ Fit dynamics. """
        X = sample_data.get_X()  # Use all samples to fit dynamics.
        U = sample_data.get_U()
        N, T, dX = X.shape
        dU = U.shape[2]

        if N==1: 
            raise ValueError("Cannot fit dynamics on 1 sample")

        self.Fm = np.zeros([T, dX, dX+dU])
        self.fv = np.zeros([T, dX])
        self.dyn_covar = np.zeros([T, dX, dX])

        it = slice(dX+dU)
        ip = slice(dX+dU, dX+dU+dX)
        # Fit dynamics wih least squares regression
        for t in range(T-1):
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]]

            mu0, Phi, m, n0 = self.prior.eval(dX,dU,xux)

            xux_mean = np.mean(xux, axis=0)
            empsig = (xux-xux_mean).T.dot(xux-xux_mean) / (N-1)
            empsig = 0.5*(empsig+empsig.T)

            sigma = (N*empsig + Phi + ((N*m)/(N+m))*np.outer(xux_mean-mu0,xux_mean-mu0))/(N+n0)
            sigma = 0.5*(sigma+sigma.T)
            sigma[it,it] = sigma[it,it]+self._hyperparams['regularization']*np.eye(dX+dU)

            Fm = (np.linalg.pinv(sigma[it, it]).dot(sigma[it, ip])).T
            fv = xux_mean[ip] - Fm.dot(xux_mean[it]);

            self.Fm[t, :, :] = Fm
            self.fv[t, :] = fv

            dyn_covar = sigma[ip,ip] - Fm.dot(sigma[it,it]).dot(Fm.T)
            self.dyn_covar[t, :, :] = 0.5*(dyn_covar+dyn_covar.T)  # Make symmetric
