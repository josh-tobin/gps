import numpy as np
from helper import *

class OnlineDynamics(object):
    def __init__(self, gamma, prior):
        self.gamma = gamma
        self.prior = prior
        self.xxt = None  # TODO: Init these
        self.mu = None
        self.sigma = None
        self.sigreg = 1e-5  # Covariance regularization (adds sigreg*eye(N))

    def update(self, prevx, prevu, curx):
        dX = self.dX
        dU = self.dU
        ix = slice(dX)
        iu = slice(dX, dX + dU)
        it = slice(dX + dU)
        ip = slice(dX + dU, dX + dU + dX)
        xu = np.r_[prevx, prevu].astype(np.float32)
        xux = np.r_[prevx, prevu, curx].astype(np.float32)

        gamma = self.gamma
        # Do a moving average update
        self.mu = self.mu * (1 - gamma) + xux * (gamma)
        self.xxt = self.xxt * (1 - gamma) + np.outer(xux, xux) * (gamma)
        self.xxt = 0.5 * (self.xxt + self.xxt.T)
        self.sigma = self.xxt - np.outer(self.mu, self.mu)

    def get_dynamics(self, t, prevx, prevu, curx, curu):
        dX = self.dX
        dU = self.dU

        ix = slice(dX)
        iu = slice(dX, dX + dU)
        it = slice(dX + dU)
        ip = slice(dX + dU, dX + dU + dX)

        mun = self.mu
        empsig = self.sigma
        N = self.empsig_N

        xu = np.r_[curx, curu].astype(np.float32)
        pxu = np.r_[prevx, prevu]
        xux = np.r_[prevx, prevu, curx]
        sigma, mun = self.prior.eval(dX, dU, xu, pxu, xux, self.sigma, self.mu, N)

        sigma[it, it] = sigma[it, it] + self.sigreg * np.eye(dX + dU)
        sigma_inv = invert_psd(sigma[it,it])

        Fm = sigma_inv.dot(sigma[it, ip]).T
        fv = mun[ip] - Fm.dot(mun[it])
        dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
        dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)  # Make symmetric
        return Fm, fv, dyn_covar


class OnlineDynamicsPrior(object):
    def __init__(self):
        pass

    def eval(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        """Default: Prior that does nothing"""
        return empsig, mun


class NNPrior(OnlineDynamicsPrior):
    def __init__(self, dyn_network, mix_strength):
        self.dyn_net = dyn_network
        self.mix_prior_strength = mix_strength
        self.sigreg = 1e-5  # Covariance regularization (adds sigreg*eye(N))

    def eval(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        F, f = self.dyn_net.getF(xu)
        nn_Phi, nnf = mix_nn_prior(F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
        sigma = (N * empsig + nn_Phi) / (N + 1)
        mun = (N * mun + np.r_[xu, F.dot(xu) + f]) / (N + 1)
        return sigma, mun


class GMMPrior(OnlineDynamicsPrior):
    def __init__(self, dynprior):
        self.dynprior = dynprior

    def eval(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        mu0, Phi, m, n0 = self.dynprior.eval(dX, dU, xux.reshape(1, dX + dU + dX))
        m = m[0][0]
        mun = (N * mun + mu0 * m) / (N + m)  # Use bias
        sigma = (N * empsig + Phi + ((N * m) / (N + m)) * np.outer(mun - mu0, mun - mu0)) / (N + n0)
        return sigma, mun


class LSQPrior(OnlineDynamicsPrior):
    def __init__(self, init_sigma, init_mu, mix_strength):
        self.dyn_init_sig = init_sigma
        self.dyn_init_mu = init_mu
        self.mix_prior_strength = mix_strength

    def eval(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        mu0, Phi = (self.dyn_init_mu, self.dyn_init_sig)
        mun = (N * mun + mu0 * self.mix_prior_strength) / (N + self.mix_prior_strength)
        sigma = (N * empsig + self.mix_prior_strength * Phi) / (
            N + self.mix_prior_strength)  # + ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
        return sigma, mun