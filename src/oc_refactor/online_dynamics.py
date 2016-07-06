import numpy as np
import cPickle

from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.utility.general_utils import finite_differences, ParallelFiniteDifferences
from helper import *
import dynamics_nn
import imp
import multiprocessing as mp

class OnlineDynamics(object):
    __metaclass__ = ClassRegistry

    def __init__(self):
        pass

    def update(self, prevx, prevu, curx):
        raise NotImplementedError("Abstract method")

    def get_dynamics(self, t, prevx, prevu, curx, curu):
        raise NotImplementedError("Abstract method")

class OnlineOutputErrorDynamics(OnlineDynamics):
    def __init__(self, gamma, prior, dX, dU, 
            covariance_scale=1e-4, N=1, sigreg=1e-5):
        self.gamma = gamma
        self.prior = prior
        self.sigreg = sigreg
        self.covariance_scale=covariance_scale
        self.dX = dX
        self.dU = dU
        self.empsig_N = N #(necessary?)
        self.x_tm2 = None # Save prev x and prev u
        self.u_tm2 = None # for dynamics model
        self.preverr = None
        self.err = None
    
    @staticmethod
    def from_config(prior, config=None):
        return OnlineOutputErrorDynamics(config['gamma'], prior, 
                config['dX'], config['dU'], 
                covariance_scale=config['covariance_scale'])

    def _calc_err(self, curx, predx):
        gamma = self.gamma
        if self.preverr is None:
            err = curx - predx
        else:
            err = (1 - gamma) * self.preverr + \
                    gamma * (curx - predx)
        return err

    def update(self, prevx, prevu, curx):
        try:
            predx = self.prior.estimate_next(
                    prevx, prevu, self.x_tm2, self.u_tm2)
        except ValueError:
            predx = curx

        tmperr = self.err
        self.err = self._calc_err(curx, predx)
        self.preverr = tmperr

    def get_dynamics(self, t, prevx, prevu, curx, curu):
        F, f = self.prior.linearize(curx, curu, prevx, prevu)
        Fm = F
        fv = f + self.err 
        dyn_covar = self.covariance_scale * np.eye(self.dX)
        return Fm, fv, dyn_covar

class OnlineGaussianDynamics(OnlineDynamics):
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
        
    @staticmethod
    def from_config(prior, config=None):
        return OnlineGaussianDynamics(config['gamma'], prior, config['dyn_init_mu'], config['dyn_init_sig'], config['dX'], config['dU'])

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

    def linearize(self, x, u, x_prev, u_prev):
        xu = np.r_[x, u]
        xu_prev = np.r_[x_prev, u_prev]
        return self.dyn_net.linearize(xu, xu_prev)

    def estimate_next(self, x, u, x_prev, u_prev):
        xu = np.r_[x, u]
        xu_prev = np.r_[x_prev, u_prev]
        return self.dyn_net.fwd_single(xu, xu_prev)


#class ParallelSampler(object):
#    def __init__(self, n_workers=6):
#        self.pool = mp.Pool(n_workers)

#    def initialize_models(self, modelfile):
#        pool.apply_async(

class ModeledPrior(OnlineDynamicsPrior):
    @staticmethod
    def from_config(modelfile, config=None):
        dynamics_params = imp.load_source('hyperparams', modelfile)
        model_agent = dynamics_params.agent['type'](
                dynamics_params.agent)
        return ModeledPrior(
                model_agent,
                mix_strength=config['mix_strength'],
                fd_eps=config['fd_eps'],
                dX_include=config['dX_include'],
                dX=config['dX'],
                dU=config['dU'],
                condition=config['condition']
        )

    def __init__(self, model_agent, mix_strength=1.0, fd_eps=1e-5,
                 dX_include=14, dX=32, dU=7, condition=0,
                 n_workers=6):
        self.model_agent = model_agent
        self.mix_prior_strength = mix_strength
        self.fd_eps = fd_eps
        self.dX_include = dX_include
        self.dX = dX
        self.dU = dU
        self.condition = condition
        #self.finite_differences = ParallelFiniteDifferences(
            #self._dynamics_function,
            #func_output_shape=(self.dX,),
            #epsilon=self.fd_eps,
        #    n_workers=n_workers
        #)

    def _dynamics_function(self, xu):
        x = xu[:self.dX_include]
        prev_eepts = xu[self.dX_include:self.dX]
        idx = len(prev_eepts) // 2
        prev_eepts = prev_eepts[:idx]
        u = xu[self.dX:self.dX+self.dU]
        #new_x = self.model_agent.step(self.condition, x, u)
        return self.model_agent.step(self.condition, x, u, prev_eepts)


    def estimate_next(self, x, u, x_prev, u_prev):
        xu = np.r_[x, u]
        return self._dynamics_function(xu)
   
    def linearize(self, x, u, x_prev, u_prev):
        xu = np.r_[x,u].astype(np.float64)
        df_dxu = finite_differences(self._dynamics_function,
                xu, (self.dX,), epsilon=self.fd_eps).T
        #df_dxu = self.finite_differences(xu)
        #df_dxu = self.finite_differences(self._dynamics_function,
        #        xu, (self.dX,), epsilon=self.fd_eps).T
        Fm = df_dxu
        fv = self._dynamics_function(xu).astype(np.float64) - Fm.dot(xu)
        return Fm, fv
    
    def mix(self, dX, dU, xu, pxu, xux, empsig, mun, N):
        F, f = self.linearize(xu[:self.dX], xu[self.dX:], pxu[:self.dX], pxu[self.dX:])
        nn_Phi, nnf = mix_prior(dX, dU, F, f, xu, strength=self.mix_prior_strength, 
                use_least_squares=False)
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
