import numpy as np

from policy import Policy


class LinearGaussianPolicy(Policy):
    """
    Time-varying linear gaussian policy.

    U = K*(x - x_hat) + u_hat + k + noise
    Where noise ~ N(0, chol_pol_covar)

    Args:
        K: T x Du x Dx
        k: T x Du
        ref: T x (Dx+Du)
        pol_covar: T x Du x Du
        chol_pol_covar: T x Du x Du
        inv_pol_covar: T x Du x Du
    """
    def __init__(self, K, k, ref, pol_covar, chol_pol_covar, inv_pol_covar):
        Policy.__init__(self)
        #TODO: Pull dimensions from somewhere else
        self.T = K.shape[0]
        self.dU = K.shape[1]
        self.dX = K.shape[2]

        self.K = K
        self.k = k
        self.ref = ref
        self.x_hat = ref[:, :self.dX]
        self.u_hat = ref[:, self.dX:self.dU]
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
        u = self.K[t].dot(x - self.x_hat[t]) + self.u_hat[t] + self.k[t]
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self, noise):
        """
        Fold x_hat, u_hat, and noise into k.

        The new k is:
        k = k + u_hat + noise - K*x_hat

        This simplifies the policy to:
        U = K*x + k

        Args:
            noise: A T x Du noise vector.
        Returns:
            k: A T x dU bias vector
        """
        k = np.zeros_like(self.k)
        for i in range(self.T):
            noise = self.chol_pol_covar[i].T.dot(noise[i])
            k[i] = self.u_hat[i] + noise - self.K[i].dot(self.x_hat[i]) + self.k[i]
        return k
