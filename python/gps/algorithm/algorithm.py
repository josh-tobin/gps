import abc
import numpy as np


class Algorithm(object):
    """Algorithm superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.traj_opt = hyperparams['traj_opt']['type'](hyperparams['traj_opt'])
        self.cost = [hyperparams['cost']['type'](hyperparams['cost'])]*hyperparams['conditions']

        self.M = hyperparams['conditions']
        self.iteration_count = 0  # Keep track of what iteration this is currently on

        # Set initial values
        init_args = hyperparams['init_traj_distr']['args']
        self.T = init_args['T']
        self.dX = init_args['dX']
        self.dU = init_args['dU']
        self.dO = self._hyperparams['dO']

    @abc.abstractmethod
    def iteration(self, sample_list):
        """ Run iteration of the algorithm. """
        raise NotImplementedError("Must be implemented in subclass")

    def _update_dynamics(self):
        """
        Instantiate dynamics objects and update prior.
        Fit dynamics to current samples
        """
        for m in range(self.M):
            if self.iteration_count >= 1:
                self.prev[m].traj_info.dynamics = self.dynamics[m].copy()
            self.cur[m].traj_info.dynamics = self.dynamics[m]
            cur_data = self.cur[m].sample_list
            self.cur[m].traj_info.dynamics.update_prior(cur_data)

            self.cur[m].traj_info.dynamics.fit(cur_data)

            init_X = cur_data.get_X()[:, 0, :]
            x0mu = np.mean(init_X, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(np.maximum(np.var(init_X, axis=0),
                self._hyperparams['initial_state_var']))

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += Phi + ((N*priorm)/(N+priorm))*np.outer(x0mu-mu0,x0mu-mu0)/(N+n0)

    def _update_trajectories(self):
        """
        Compute new linear gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [self.cur[m].traj_distr for m in range(self.M)]
        for m in range(self.M):
            pol_info = self.cur[m].pol_info if 'pol_info' in dir(self.cur[m]) else None
            self.new_traj_distr[m], self.eta[m] = self.traj_opt.update(
                    self.T, self.cur[m].step_mult, self.eta[m],
                    self.cur[m].traj_info, self.new_traj_distr[m],
                    pol_info)

    def _eval_cost(self, m):
        """
        Evaluate costs for all samples for a condition

        Args:
            m: Condition
        """
        # Constants.
        T = self.T
        dX = self.dX
        dU = self.dU
        N = len(self.cur[m].sample_list)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, dX + dU))
        Cm = np.zeros((N, T, dX + dU, dX + dU))
        for n in range(N):
            sample = self.cur[m].sample_list[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[m].eval(sample)
            cc[n, :] = l
            cs[n, :] = l
            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]  # T x (X+U)
            Cm[n, :, :, :] = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

            # Adjust for expanding cost around a sample.
            X = sample.get_X()
            U = sample.get_U()
            yhat = np.c_[X, U]
            rdiff = -yhat  # T x (X+U)
            rdiff_expand = np.expand_dims(rdiff, axis=2)  # T x (X+U) x 1
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)  # T x (X+U)
            cc[n, :] += np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] += cv_update

        self.cur[m].traj_info.cc = np.mean(cc, 0)  # Costs. Average over samples
        self.cur[m].traj_info.cv = np.mean(cv, 0)  # Cost, 1st deriv
        self.cur[m].traj_info.Cm = np.mean(Cm, 0)  # Cost, 2nd deriv
        self.cur[m].cs = cs
