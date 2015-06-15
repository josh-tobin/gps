#!/usr/bin/env python

class LRDynamics(Dynamics):
    """Dynamics with linear regression, with constant prior.

    """
    def __init__(self,hyperparams,sample_data):
        Dynamics.__init__(self, hyperparams, sample_data)

    def update_prior(self):
        """ Update dynamics prior. """
        # Nothing to do - constant prior.
        pass

    def eval_prior(self):
        raise NotImplementedError("TODO");

    def fit(self, sample_idx):
        """ Fit dynamics. """
        X = self._sample_data.get_X(sample_idx)
        U = self._sample_data.get_U(sample_idx)

        dX = X.shape[-1]
        dU = U.shape[-1]

        sig = np.eye(dX+dU)
        raise NotImplementedError("TODO - Fit dynamics");
