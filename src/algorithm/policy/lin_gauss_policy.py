import numpy as np

from policy import Policy

class LinearGaussianPolicy(Policy):
    """
    Time-varying linear gaussian policy.

    U = K*(x - x_hat) + u_hat + k + chol_pol_covar*noise

    Args:
        K: T x Du x Dx
        k: T x Du
        x_hat: T x Dx
        u_hat: T x Du
        chol_pol_covar: T x Du x Du
    """
    def __init__(self, K, k, x_hat, u_hat, chol_pol_covar):
        Policy.__init__(self)
        self.K = K
        self.k = k
        self.x_hat = x_hat
        self.u_hat = u_hat
        self.chol_pol_covar = chol_pol_covar
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
        # Compute noise - this is executed once and thrown away
        if noise is None:
            noise = np.random.randn(self.T, self.dU)
        noise = self.chol_pol_covar.T.dot(noise)
        return self.u_hat + noise - self.K.dot(self.x_hat) + self.k
