import copy
import numpy as np
import logging

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.config import alg_traj_opt
from gps.utility.general_utils import bundletype


LOGGER = logging.getLogger(__name__)

# Set up an object to bundle variables
ITERATION_VARS = ['sample_list', 'traj_info', 'traj_distr', 'cs',
                  'step_change', 'mispred_std', 'polkl', 'step_mult']
IterationData = bundletype('ItrData', ITERATION_VARS)

# Note: last_kl_step isn't used in this alg, but is used in others (alg_badmm)
TRAJINFO_VARS = ['dynamics', 'x0mu', 'x0sigma', 'cc', 'cv', 'Cm', 'last_kl_step']
TrajectoryInfo = bundletype('TrajectoryInfo', TRAJINFO_VARS)


class AlgorithmTrajOpt(Algorithm):
    """Sample-based trajectory optimization.

    """

    def __init__(self, hyperparams):
        config = copy.deepcopy(alg_traj_opt)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        # Keep 1 iteration data for each condition
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        init_args = hyperparams['init_traj_distr']['args']
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
        self._update_dynamics()

        self._update_step_size()  # KL Divergence step size

        # Run inner loop to compute new policies under new dynamics and step size
        for inner_itr in range(self._hyperparams['inner_iterations']):
            self._update_trajectories()

        self._advance_iteration_variables()

    # TODO - can this go in super class
    def _update_step_size(self):
        """ Evaluate costs on samples, adjusts step size """
        # Evaluate cost function for all conditions and samples
        for m in range(self.M):
            self._eval_cost(m)

        for m in range(self.M):  # m = condition
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                # Evaluate cost and adjust step size relative to the previous iteration.
                self._stepadjust(m)

    def _stepadjust(self, m):
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

    # TODO - move to super class
    def _advance_iteration_variables(self):
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
        delattr(self, 'new_traj_distr')
