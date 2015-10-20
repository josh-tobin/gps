import copy
import numpy as np
import logging

from gps.algorithm.config import alg_traj_opt
from gps.algorithm.algorithm import Algorithm
from gps.utility.general_utils import bundletype


LOGGER = logging.getLogger(__name__)

# Set up an object to bundle variables
ITERATION_VARS = ['sample_list', 'traj_info', 'traj_distr', 'cs',
                  'step_change', 'mispred_std', 'polkl', 'step_mult']
IterationData = bundletype('ItrData', ITERATION_VARS)

TRAJINFO_VARS = ['dynamics', 'x0mu', 'x0sigma', 'cc', 'cv', 'Cm']
TrajectoryInfo = bundletype('TrajectoryInfo', TRAJINFO_VARS)


class AlgorithmTrajOpt(Algorithm):
    """Sample-based trajectory optimization.

    """

    def __init__(self, hyperparams):
        config = copy.deepcopy(alg_traj_opt)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        # Construct objects
        self.M = self._hyperparams['conditions']

        self.iteration_count = 0  # Keep track of what iteration this is currently on

        # Keep 1 iteration data for each condition
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Set initial values
        init_args = self._hyperparams['init_traj_distr']['args']
        self.T = init_args['T']
        self.dX = init_args['dX']
        self.dU = init_args['dU']

        self.dynamics = [None]*self.M
        for m in range(self.M):
            self.cur[m].traj_distr = self._hyperparams['init_traj_distr']['type'](**init_args)
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].step_mult = 1.0
            self.dynamics[m] = self._hyperparams['dynamics']['type'](
                self._hyperparams['dynamics'])
        self.eta = [1.0]*self.M


    def iteration(self, sample_lists):
        """
        Run iteration of LQR.
        Args:
            sample_lists: List of sample_list objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        # Update dynamics model using all sample.
        self.update_dynamics()

        self.update_step_size()  # KL Divergence step size

        # Run inner loop to compute new policies under new dynamics and step size
        self.update_trajectories()

        self.advance_iteration_variables()

    def update_dynamics(self):
        """
        Instantiate dynamics objects and update prior.
        Fit dynamics to current samples
        """
        for m in range(self.M):
            self.cur[m].traj_info.dynamics = self.dynamics[m]
            cur_data = self.cur[m].sample_list
            self.cur[m].traj_info.dynamics.update_prior(cur_data)

            self.cur[m].traj_info.dynamics.fit(cur_data)

            init_X = cur_data.get_X()[:, 0, :]
            x0mu = np.mean(init_X, axis=0)
            self.cur[m].traj_info.x0mu = x0mu
            self.cur[m].traj_info.x0sigma = np.diag(np.maximum( np.var(init_X, axis=0),
                    self._hyperparams['initial_state_var']))

            prior = self.cur[m].traj_info.dynamics.get_prior()
            if prior:
                mu0, Phi, priorm, n0 = prior.initial_state()
                N = len(cur_data)
                self.cur[m].traj_info.x0sigma += Phi + ((N*priorm)/(N+priorm))*np.outer(x0mu-mu0,x0mu-mu0)/(N+n0)

    def update_step_size(self):
        """ Evaluate costs on samples, adjusts step size """
        # Evaluate cost function for all conditions and samples
        for m in range(self.M):
            self.eval_cost(m)

        for m in range(self.M):  # m = condition
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                # Evaluate cost and adjust step size relative to the previous iteration.
                self.stepadjust(m)

    def update_trajectories(self):
        """
        Compute new linear gaussian controllers.
        """
        self.new_traj_distr = [None]*self.M
        for inner_itr in range(self._hyperparams['inner_iterations']):
            for m in range(self.M):
                self.new_traj_distr[m], self.eta[m] = self.traj_opt.update(
                        self.T, self.cur[m].step_mult, self.eta[m],
                        self.cur[m].traj_info, self.cur[m].traj_distr)

    def stepadjust(self, m):
        """
        Calculate new step sizes.

        Args:
            m: Condition
        """
        # No policy by default.
        polkl = np.zeros(self.T)

        # Compute values under Laplace approximation.
        # This is the policy that the previous samples were actually drawn from
        # under the dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(self.prev[m].traj_distr, self.prev[m].traj_info)
        # This is the policy that we just used under the dynamics that were
        # estimated from the previous samples (so this is the cost we thought we
        # would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(self.cur[m].traj_distr, self.prev[m].traj_info)

        # This is the actual cost we have under the current trajectory based on the
        # latest samples.
        new_actual_laplace_obj = self.traj_opt.estimate_cost(self.cur[m].traj_distr, self.cur[m].traj_info)

        # Measure the entropy of the current trajectory (for printout).
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :])))

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f', ent, previous_mc_obj, new_mc_obj)

        # Compute misprediction vs Monte-Carlo score.
        mispred_std = np.abs(np.sum(new_actual_laplace_obj) - new_mc_obj) / \
            max(np.std(np.sum(self.cur[m].cs, axis=1), axis=0), 1.0)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - np.sum(new_actual_laplace_obj)

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f', np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f', np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f', np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Predicted/actual improvement: %f / %f', predicted_impr, actual_impr)

        # model improvement as: I = predicted_dI * KL + penalty * KL^2
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2)
        # optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL => KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL
        # therefore, the new multiplier is given by pred/2*(pred-act)
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[m].step_mult, self._hyperparams['max_step_mult']),
                          self._hyperparams['min_step_mult'])
        step_change = new_step / self.cur[m].step_mult
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

        self.cur[m].step_change = step_change
        self.cur[m].mispred_std = mispred_std
        self.cur[m].polkl = polkl

    def eval_cost(self, m):
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

    def advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev'.
        Advance iteration counter
        """
        self.iteration_count += 1
        self.prev = self.cur
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].traj_distr = self.new_traj_distr[m]

