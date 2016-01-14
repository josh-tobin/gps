import abc
import copy
import numpy as np

from gps.algorithm.config import alg


class Algorithm(object):
    """
    Algorithm superclass
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        config = copy.deepcopy(alg)
        config.update(hyperparams)
        self._hyperparams = config

        self.M = hyperparams['conditions']
        self.iteration_count = 0  # Keep track of what iteration this is currently on.

        self.traj_opt = hyperparams['traj_opt']['type'](hyperparams['traj_opt'])
        self.cost = [hyperparams['cost']['type'](hyperparams['cost']) for _ in range(self.M)]

        # Grab a few values from the agent.
        agent = self._hyperparams['agent']
        self.T = self._hyperparams['T'] = agent.T
        self.dU = self._hyperparams['dU'] = agent.dU
        self.dX = self._hyperparams['dX'] = agent.dX
        self.dO = self._hyperparams['dO'] = agent.dO

        init_traj_distr = config['init_traj_distr']
        init_traj_distr['x0'] = agent.x0
        init_traj_distr['dX'] = agent.dX
        init_traj_distr['dU'] = agent.dU
        del self._hyperparams['agent']  # Don't want to pickle this.

    @abc.abstractmethod
    def iteration(self, sample_list):
        """
        Run iteration of the algorithm.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior. Fit dynamics to current samples.
        """
        for m in range(self.M):
            if self.iteration_count >= 1:
                self.prev[m].traj_info.dynamics = self.cur[m].traj_info.dynamics.copy()
            cur_data = self.cur[m].sample_list
            self.cur[m].traj_info.dynamics.update_prior(cur_data)

            self.cur[m].traj_info.dynamics.fit(cur_data)

            init_X = cur_data.get_X()[:,0,:]
            x0mu = np.mean(init_X, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(np.maximum(np.var(init_X, axis=0),
                self._hyperparams['initial_state_var']))

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += Phi + (N*priorm) / (N+priorm) * \
                        np.outer(x0mu-mu0, x0mu-mu0) / (N+n0)

    def _update_trajectories(self):
        """
        Compute new linear Gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [self.cur[m].traj_distr for m in range(self.M)]
        for m in range(self.M):
            self.new_traj_distr[m], self.cur[m].eta = self.traj_opt.update(m, self)

    def _eval_cost(self, m):
        """
        Evaluate costs for all samples for a condition.
        Args:
            m: Condition to evaluate cost on.
        """
        # Constants.
        T, dX, dU = self.T, self.dX, self.dU
        N = len(self.cur[m].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX+dU))
        Cm = np.zeros((N, T, dX+dU, dX+dU))
        for n in range(N):
            sample = self.cur[m].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[m].eval(sample)
            cc[n,:] = l
            cs[n,:] = l

            # Assemble matrix and vector.
            cv[n,:,:] = np.c_[lx, lu]
            Cm[n,:,:,:] = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat
            rdiff_expand = np.expand_dims(rdiff, axis=2)
            cv_update = np.sum(Cm[n,:,:,:] * rdiff_expand, axis=1)
            cc[n,:] += np.sum(rdiff * cv[n,:,:], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n,:,:] += cv_update

        # Fill in cost estimate.
        self.cur[m].traj_info.cc = np.mean(cc, 0)  # Constant term (scalar).
        self.cur[m].traj_info.cv = np.mean(cv, 0)  # Linear term (vector).
        self.cur[m].traj_info.Cm = np.mean(Cm, 0)  # Quadratic term (matrix).

        self.cur[m].cs = cs # True value of cost.
