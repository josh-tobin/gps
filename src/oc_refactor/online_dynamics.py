import numpy as np
import cPickle

from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from helper import *
import dynamics_nn


class OnlineDynamics(object):
    def __init__(self, gamma, prior, init_mu, init_sigma, dX, dU, N=1, sigreg=1e-5):
        self.gamma = gamma
        self.prior = prior
        self.sigreg = sigreg # Covariance regularization (adds sigreg*eye(N))
        self.dX = dX
        self.dU = dU
        self.empsig_N = N

        # Initial values
        self.mu = init_mu
        self.sigma = init_sigma
        self.xxt = init_sigma + np.outer(self.mu, self.mu)

    def update(self, prevx, prevu, curx):
        """ Perform a moving average update on the current dynamics """
        dX, dU = self.dX, self.dU
        dX = prevx.shape[0]
        dU = prevu.shape[0]
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
        """ 
        Compute F, f - the linear dynamics where next_x = F*[curx, curu] + f 
        """
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

        # Mix and add regularization
        sigma, mun = self.prior.mix(dX, dU, xu, pxu, xux, self.sigma, self.mu, N)
        sigma[it, it] = sigma[it, it] + self.sigreg * np.eye(dX + dU)
        sigma_inv = invert_psd(sigma[it, it])

        # Solve normal equations to get dynamics.
        Fm = sigma_inv.dot(sigma[it, ip]).T
        fv = mun[ip] - Fm.dot(mun[it])
        dyn_covar = sigma[ip, ip] - Fm.dot(sigma[it, it]).dot(Fm.T)
        dyn_covar = 0.5 * (dyn_covar + dyn_covar.T)  # Guarantee symmetric

        return Fm, fv, dyn_covar


class OnlineDynamicsPrior(object):
    __metaclass__ = ClassRegistry
    def __init__(self):
        pass

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        raise NotImplementedError()


class NoPrior(OnlineDynamicsPrior):
    @staticmethod
    def from_config(config=None):
        return NoPrior()

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        """Default: Prior that does nothing"""
        return empsig, mun


class NNPrior(OnlineDynamicsPrior):
    @staticmethod
    def from_config(netname, config=None):
        dyn_net = dynamics_nn.unpickle_net(netname)
        dyn_net.update(stage='test')
        dyn_net.init_functions(output_blob='acc')
        return NNPrior(dyn_net, config['mix_prior_strength'])

    def __init__(self, dyn_net, mix_strength=1.0):
        self.dyn_net = dyn_net
        self.mix_prior_strength = mix_strength

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        F, f = self.dyn_net.linearize(xu, pxu)
        nn_Phi, nnf = mix_prior(dX, dU, F, f, xu, strength=self.mix_prior_strength, use_least_squares=False)
        sigma = (N * empsig + nn_Phi) / (N + 1)
        mun = (N * mun + np.r_[xu, F.dot(xu) + f]) / (N + 1)
        return sigma, mun


class GMMPrior(OnlineDynamicsPrior):
    @staticmethod
    def from_config(controllerfile, config=None):
        cond = config['condition']
        with open(controllerfile + '_' + str(cond)) as f:
            controller_dict = cPickle.load(f)
            gmm = controller_dict['gmm']
            dynprior = DynamicsPriorGMM({})
            dynprior.gmm = gmm
        return GMMPrior(dynprior)

    def __init__(self, dynprior):
        self.dynprior = dynprior

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        mu0, Phi, m, n0 = self.dynprior.eval(dX, dU, xux.reshape(1, dX + dU + dX))
        m = m[0][0]
        mun = (N * mun + mu0 * m) / (N + m)  # Use bias
        sigma = (N * empsig + Phi + ((N * m) / (N + m)) * np.outer(mun - mu0, mun - mu0)) / (N + n0)
        return sigma, mun


class LSQPrior(OnlineDynamicsPrior):
    @staticmethod
    def from_config(config=None):
        return LSQPrior(config['dyn_init_sigma'], config['dyn_init_mu'], config['mix_prior_strength'])

    def __init__(self, init_sigma, init_mu, mix_strength=1.0):
        self.dyn_init_sig = init_sigma
        self.dyn_init_mu = init_mu
        self.mix_prior_strength = mix_strength

    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        mu0, Phi = (self.dyn_init_mu, self.dyn_init_sig)
        mun = (N * mun + mu0 * self.mix_prior_strength) / (N + self.mix_prior_strength)
        sigma = (N * empsig + self.mix_prior_strength * Phi) / (
            N + self.mix_prior_strength)  # + ((N*m)/(N+m))*np.outer(mun-mu0,mun-mu0))/(N+n0)
        return sigma, mun