import numpy as np

from dynamics import Dynamics


class DynamicsLR(Dynamics):
    """Dynamics with linear regression, with constant prior.

    """
    def __init__(self,hyperparams,sample_data):
        Dynamics.__init__(self, hyperparams, sample_data)

        self._ref_traj_X = np.zeros(T, dX)
        self._ref_traj_U = np.zeros(T, dU)

    def update_prior(self):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def fit(self):
        """ Fit dynamics. """
        X = self._sample_data.get_X()
        U = self._sample_data.get_U()

        N, T, dX = X.shape
        dU = U.shape[2]

        Fd = np.zeros(T, dX+dU, dX)

        # Subtract reference trajectory
        X = np.asarray([X[i, :, :] - self._ref_traj for i in range(N)])
        U = np.asarray([U[i, :, :] - self._ref_traj for i in range(N)])

        # Fit dynamics wih least squares regression
        # TODO - deal with fc (add ones and slice result?)
        for t in range(T-1):
            Fd[t,:,:], _, _, _ = np.linalg.lstsq(np.c_[X[:,t,:],U[:,t,:]], X[:,t+1,:])

        # TODO - leave last time step as zeros?

        dyn_covar = np.eye(dX+dU)
        raise NotImplementedError("TODO - Fit dynamics")
