from dynamics import Dynamics

class DynamicsLR(Dynamics):
    """Dynamics with linear regression, with constant prior.

    """
    def __init__(self,hyperparams,sample_data):
        Dynamics.__init__(self, hyperparams, sample_data)
        super(DynamicsLR, self).__init__(hyperparams, sample_data)

    def update_prior(self):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def fit(self):
        """ Fit dynamics. """
        X = self._sample_data.get_X()
        U = self._sample_data.get_U()

        dX = X.shape[-1]
        dU = U.shape[-1]

        sig = np.eye(dX+dU)
        raise NotImplementedError("TODO - Fit dynamics")
