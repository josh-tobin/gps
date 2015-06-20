import numpy as np

from policy import Policy

class LinearGaussianPolicy(Policy):
    def __init__(self, K, k, x_hat, u_hat, chol_pol_covar):
        Policy.__init__(self)
        self.K = K
        self.k = k
        self.x_hat = x_hat
        self.u_hat = u_hat
        self.chol_pol_covar = chol_pol_covar
        self.T = K.shape[0]
        self.dU = k.shape[1]

    def act(self, x, obs, t):
        u = self.K[t].dot(x - self.x_hat[t]) + self.u_hat[t] + self.k[t]
        noise = np.random.randn(1, self.dU)
        u += self.chol_pol_covar[t].T.dot(noise)
        return u

    def fold_k(self):
        """
        Fold x_hat, u_hat, and covariance into k
        Returns:
            k: A T x dU bias vector
        """
        # Compute noise - this is executed once and thrown away
        noise = self.chol_pol_covar.T.dot(np.random.randn(self.T, self.dU))
        return self.u_hat + noise - self.K.dot(self.x_hat) + self.k
