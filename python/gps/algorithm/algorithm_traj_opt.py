import numpy as np
import logging
import copy

from algorithm import Algorithm
from config import alg_traj_opt
from dynamics.dynamics_lr import DynamicsLR
from traj_opt.traj_info import TrajectoryInfo

LOGGER = logging.getLogger(__name__)


class AlgorithmTrajOpt(Algorithm):
    """Sample-based trajectory optimization.

    """

    def __init__(self, hyperparams, sample_data):
        config = copy.deepcopy(alg_traj_opt)
        config.update(hyperparams)
        Algorithm.__init__(self, config, sample_data)

        # Construct objects
        self.M = self._hyperparams['conditions']

        self.iteration_count = 0  # Keep track of what iteration this is currently on
        # TODO: Remove. This is very hacky
        # List of variables updated from iteration to iteration
        self.iteration_vars = ['sample_data', 'trajinfo', 'traj_distr', 'cs',
                               'step_change', 'mispred_std', 'polkl', 'step_mult']

        # TODO: Remove. This is very hacky
        for varname in self.iteration_vars:
            setattr(self, 'cur_' + varname, [None]*self.M)
            setattr(self, 'prev_' + varname, [None]*self.M)

        # Set initial values
        init_args = self._hyperparams['init_traj_distr']['args']
        init_args['dX'] = sample_data.dX
        init_args['dU'] = sample_data.dU
        init_args['T'] = sample_data.T
        self.cur_traj_distr = [self._hyperparams['init_traj_distr']['type'](**init_args)]*self.M

        hyperparams['init_traj_distr']
        self.cur_trajinfo = [TrajectoryInfo() for _ in range(self.M)]
        self.cur_step_mult = [1.0]*self.M
        self.eta = [1.0]*self.M

    def iteration(self, sample_data):
        """
        Run iteration of LQR.
        Args:
            sample_data: List of sample_data for each condition.
        """
        self.cur_sample_data = sample_data

        # Update dynamics model using all sample.
        self.update_dynamics_prior()
        self.fit_dynamics()

        self.eval_costs()
        self.update_step_size()  # KL Divergence step size

        # Run inner loop to compute new policies under new dynamics and step size
        self.update_trajectories()

        self.advance_iteration_variables()

    def update_dynamics_prior(self):
        """
        Instantiate dynamics objects and update prior.
        """
        for m in range(self.M):
            self.cur_trajinfo[m].dynamics = DynamicsLR(self._hyperparams['dynamics'], self.cur_sample_data[m])
            self.cur_trajinfo[m].dynamics.update_prior()
            self.cur_trajinfo[m].x0mu = np.mean(self.cur_sample_data[m].get_X()[:, 0, :], axis=0)
            self.cur_trajinfo[m].x0sigma = \
                self.cur_sample_data[m].num_samples() * \
                np.diag(np.maximum(
                    np.var(self.cur_sample_data[m].get_X()[:, 0, :], axis=0),
                    self._hyperparams['initial_state_var']))

    def fit_dynamics(self):
        """ Fit linear dynamics to samples """
        for m in range(self.M):
            # TODO: Set samples in dynamics object
            self.cur_trajinfo[m].dynamics.fit()

    def update_step_size(self):
        """ Evaluate costs on samples, adjusts step size """
        # Evaluate cost function.
        for m in range(self.M):  # m = condition
            if self.iteration_count >= 1 and self.prev_sample_data[m]:
                # Evaluate cost and adjust step size relative to the previous iteration.
                self.stepadjust(m)

    def update_trajectories(self):
        """
        Compute new linear gaussian controllers.
        """
        #TODO: Only thing that gets update between loops is eta??
        self.new_traj_distr = [None]*self.M  # Hack to get around cur/prev being automatically updated in advance_iteration_variables
        for inner_itr in range(self._hyperparams['inner_iterations']):
            for m in range(self.M):
                new_traj, self.eta[m] = self.traj_opt.update(self.cur_sample_data[m].T, self.cur_step_mult[m], self.eta[m], self.cur_trajinfo[m], self.cur_traj_distr[m])
                self.new_traj_distr[m] = new_traj

    def stepadjust(self, m):
        """
        Calculate new step sizes.

        Args:
            m: Condition
        """
        T = self.cur_sample_data[m].get_samples()[0].T
        # No policy by default.
        polkl = np.zeros(T)

        # Compute values under Laplace approximation.
        # This is the policy that the previous samples were actually drawn from
        # under the dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(self.prev_traj_distr[m], self.prev_trajinfo[m])
        # This is the policy that we just used under the dynamics that were
        # estimated from the previous samples (so this is the cost we thought we
        # would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(self.cur_traj_distr[m], self.prev_trajinfo[m])

        # This is the actual cost we have under the current trajectory based on the
        # latest samples.
        new_actual_laplace_obj = self.traj_opt.estimate_cost(self.cur_traj_distr[m], self.cur_trajinfo[m])

        # Measure the entropy of the current trajectory (for printout).
        ent = 0
        for t in range(T):
            ent = ent + np.sum(np.log(np.diag(self.cur_traj_distr[m].chol_pol_covar[t, :, :])))

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev_cs[m], axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur_cs[m], axis=1), axis=0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f', ent, previous_mc_obj, new_mc_obj)

        # Compute misprediction vs Monte-Carlo score.
        mispred_std = np.abs(np.sum(new_actual_laplace_obj) - new_mc_obj) / \
            max(np.std(np.sum(self.cur_cs[m], axis=1), axis=0), 1.0)

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
        new_mult = predicted_impr / (2 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur_step_mult[m], self._hyperparams['max_step_mult']),
                          self._hyperparams['min_step_mult'])
        step_change = new_step / self.cur_step_mult[m]
        self.cur_step_mult[m] = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

        self.cur_step_change[m] = step_change
        self.cur_mispred_std[m] = mispred_std
        self.cur_polkl[m] = polkl

    def eval_costs(self):
        """
        Evaluate costs for all conditions and samples.
        """
        for m in range(self.M):
            self.eval_cost(m)

    def eval_cost(self, m):
        """
        Evaluate costs for all samples for a condition

        Args:
            m: Condition
        """
        samples = self.cur_sample_data[m].get_samples()
        # Constants.
        Dx = samples[0].dX
        Du = samples[0].dU
        T = samples[0].T
        N = len(samples)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, Dx + Du))
        Cm = np.zeros((N, T, Dx + Du, Dx + Du))
        for n in range(N):
            sample = samples[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.cost[m].eval(sample)
            cc[n, :] = l
            cs[n, :] = l
            # Assemble matrix and vector.
            cv[n, :, :] = np.c_[lx, lu]  # T x (X+U)
            Cm[n, :, :, :] = np.concatenate((np.c_[lxx, np.transpose(lux, [0, 2, 1])], np.c_[lux, luu]), axis=1)

            # Adjust for expanding cost around a sample.
            yhat = np.c_[sample.get_X(), sample.get_U()]
            rdiff = -yhat  # T x (X+U)
            rdiff_expand = np.expand_dims(rdiff, axis=2)  # T x (X+U) x 1
            cv_update = np.sum(Cm[n, :, :, :] * rdiff_expand, axis=1)  # T x (X+U)
            cc[n, :] = cc[n, :] + np.sum(rdiff * cv[n, :, :], axis=1) + 0.5 * np.sum(rdiff * cv_update, axis=1)
            cv[n, :, :] = cv[n, :, :] + cv_update

        self.cur_trajinfo[m].cc = np.mean(cc, 0)  # Costs. Average over samples
        self.cur_trajinfo[m].cv = np.mean(cv, 0)  # Cost, 1st deriv
        self.cur_trajinfo[m].Cm = np.mean(Cm, 0)  # Cost, 2nd deriv
        self.cur_cs[m] = cs
        # TODO: Implement policy sample costs

    def advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev'.
        Advance iteration counter
        """
        self.iteration_count += 1
        # TODO: Remove. This is very hacky
        for varname in self.iteration_vars:
            setattr(self, 'prev_' + varname, getattr(self, 'cur_' + varname))
            setattr(self, 'cur_' + varname, [None]*self.M)
        self.cur_trajinfo = [TrajectoryInfo() for _ in range(self.M)]
        self.cur_step_mult = self.prev_step_mult
        self.cur_traj_distr = self.new_traj_distr  #TODO: Hack - clear this up once it runs.
