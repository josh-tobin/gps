import numpy as np

from policy import Policy


class LinearGaussianPolicy(Policy):
    """
    Time-varying linear gaussian policy.

    U = K*x + k + noise
    Where noise ~ N(0, chol_pol_covar)

    Args:
        K: T x Du x Dx
        k: T x Du
        pol_covar: T x Du x Du
        chol_pol_covar: T x Du x Du
        inv_pol_covar: T x Du x Du
    """
    def __init__(self, K, k, pol_covar, chol_pol_covar, inv_pol_covar):
        Policy.__init__(self)
        #TODO: Pull dimensions from somewhere else
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        self.K = K
        self.k = k
        self.pol_covar = pol_covar
        self.chol_pol_covar = chol_pol_covar
        self.inv_pol_covar = inv_pol_covar
        self.T = K.shape[0]
        self.dU = k.shape[1]

    def act(self, x, obs, t, noise=None):
        """
        Return an action for a state.

        Args:
            x: State vector
            obs: Observation vector
            t: timestep
            noise: Action noise vector. This will be scaled by the variance.
        """
        u = self.K[t].dot(x) + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self, noise):
        """
        Fold noise into k.

        Args:
            noise: A T x Du noise vector with ~mean 0 and variance 1
        Returns:
            k: A T x dU bias vector
        """
        k = np.zeros_like(self.k)
        for i in range(self.T):
            scaled_noise = self.chol_pol_covar[i].T.dot(noise[i])
            k[i] = scaled_noise + self.k[i]
        return k

    def nans_like(self):
        """
        Returns:
            A new LinearGaussianPolicy object with the same dimensions but
        all values filled with nan.
        """
        # TODO: Consider using deepcopy instead of this
        policy = LinearGaussianPolicy(
            np.zeros_like(self.K),
            np.zeros_like(self.k),
            np.zeros_like(self.pol_covar),
            np.zeros_like(self.chol_pol_covar),
            np.zeros_like(self.inv_pol_covar)
        )
        policy.K.fill(np.nan)
        policy.k.fill(np.nan)
        policy.pol_covar.fill(np.nan)
        policy.chol_pol_covar.fill(np.nan)
        policy.inv_pol_covar.fill(np.nan)
        return policy
