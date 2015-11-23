import copy
import numpy as np
import scipy as sp
import logging

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import estimate_moments, gauss_fit_joint_prior
from gps.algorithm.config import alg_badmm
from gps.utility.general_utils import bundletype


LOGGER = logging.getLogger(__name__)

# Set up objects to bundle variables
ITERATION_VARS = ['sample_list', 'traj_info', 'pol_info', 'traj_distr', 'cs',
                  'step_change', 'mispred_std', 'pol_kl', 'step_mult']
IterationData = bundletype('ItrData', ITERATION_VARS)

TRAJINFO_VARS = ['dynamics', 'x0mu', 'x0sigma', 'cc', 'cv', 'Cm', 'last_kl_step']
TrajectoryInfo = bundletype('TrajectoryInfo', TRAJINFO_VARS)

POLINFO_VARS = ['lambda_k', 'lambda_K', 'pol_wt', 'pol_mu', 'pol_sig',
                'pol_K', 'pol_k', 'pol_S', 'chol_pol_S', 'prev_kl']
PolicyInfo = bundletype('PolicyInfo', POLINFO_VARS)


class AlgorithmBADMM(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with BADMM.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(alg_badmm)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        # Construct objects
        self.M = self._hyperparams['conditions']

        self.iteration_count = 0  # Keep track of what iteration this is currently on

        # Keep current and previous iteration data for each condition
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        # Set initial values
        init_args = self._hyperparams['init_traj_distr']['args']
        self.T = init_args['T']
        self.dX = init_args['dX']
        self.dU = init_args['dU']
        self.dO = self._hyperparams['dO']

        self.dynamics = [None]*self.M
        for m in range(self.M):
            self.cur[m].traj_distr = self._hyperparams['init_traj_distr']['type'](**init_args)
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.last_kl_step = float('inf')
            pol_info = PolicyInfo()
            pol_info.lambda_k = np.zeros((self.T, self.dU))
            pol_info.lambda_K = np.zeros((self.T, self.dU, self.dX))
            pol_info.pol_wt = self._hyperparams['init_pol_wt'] * np.ones(self.T)
            pol_info.pol_K = np.zeros((self.T, self.dU, self.dX))
            pol_info.pol_k = np.zeros((self.T, self.dU))
            pol_info.pol_S = np.zeros((self.T, self.dU, self.dU))
            pol_info.chol_pol_S = np.zeros((self.T, self.dU, self.dU))
            self.cur[m].pol_info = pol_info
            self.cur[m].step_mult = 1.0
            self.dynamics[m] = self._hyperparams['dynamics']['type'](self._hyperparams['dynamics'])
        self.eta = [1.0]*self.M

        self.policy_opt = self._hyperparams['policy_opt']['type'](self._hyperparams['policy_opt'], self.dO, self.dU)
        #TODO: policy prior GMM, for now just constant prior
        self.policy_prior = self._hyperparams['policy_prior']['type'](self._hyperparams['policy_prior'])

    def iteration(self, sample_lists):
        """
        Run iteration of BADMM-based guided policy search.
        Args:
            sample_lists: List of sample_list objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        self._set_interp_values()

        # Update dynamics model using all sample.
        self._update_dynamics()

        self._update_step_size()  # KL Divergence step size

        # Run inner loop to compute new policies under new dynamics and step size
        for inner_itr in range(self._hyperparams['inner_iterations']):
            #TODO: could start from init controller
            if self.iteration_count > 0 or inner_itr > 0:
                self._update_policy()  # Update the policy.
            for m in range(self.M):
                self._update_policy_fit(m)  # Update policy priors.
            if self.iteration_count > 0 or inner_itr > 0:
                step = (inner_itr == self._hyperparams['inner_iterations'] - 1)
                for m in range(self.M):
                    self._policy_dual_step(m, step=step)  # Update dual variables.
            self._update_trajectories()

        self._advance_iteration_variables()

    def _set_interp_values(self):
        """
        Use iteration-based interpolation to set values of some parameters.
        """
        # Compute temporal interpolation value.
        t = min((self.iteration_count + 1.0)/(self._hyperparams['iterations'] - 1), 1)
        # Perform iteration-based interpolation of entropy penalty for policy.
        if type(self._hyperparams['ent_reg_schedule']) in (int, float):
            self.policy_opt._hyperparams['ent_reg'] = self._hyperparams['ent_reg_schedule']
        else:
            sch = self._hyperparams['ent_reg_schedule']
            self.policy_opt._hyperparams['ent_reg'] = np.exp(
                    np.interp(t, np.linspace(0, 1, num=len(sch)), np.log(sch)))
        # Perform iteration-based interpolation of Lagrange multiplier step.
        if type(self._hyperparams['lg_step_schedule']) in (int, float):
            self._hyperparams['lg_step'] = self._hyperparams['lg_step_schedule']
        else:
            sch = self._hyperparams['lg_step_schedule']
            self._hyperparams['lg_step'] = np.exp(
                    np.interp(t, np.linspace(0, 1, num=len(sch)), np.log(sch)))

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

    def _update_step_size(self):
        """ Evaluate costs on samples, adjusts step size """
        # Evaluate cost function for all conditions and samples
        for m in range(self.M):
            self._update_policy_fit(m, init=True)
            self._eval_cost(m)

        for m in range(self.M):  # m = condition
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                # Evaluate cost and adjust step size relative to the previous iteration.
                self._stepadjust(m)

    def _update_policy(self):
        """
        Compute the new policy.
        """
        dX, dU, dO, T = self.dX, self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        tgt_mu, tgt_prc, tgt_wt = np.zeros((0, T, dU)), np.zeros((0, T, dU, dU)), np.zeros((0, T))
        obs_data = np.zeros((0, T, dO))
        for m in range(self.M):
            #TODO: handle synthetic samples, for now just assuming no
            #      synthetic samples and using all samples
            samples = self.cur[m].sample_list
            X, U = samples.get_X(), samples.get_U()
            N = len(samples)
            traj, pol_info, traj_info = self.cur[m].traj_distr, self.cur[m].pol_info, self.cur[m].traj_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:,t,:,:] = np.tile(traj.inv_pol_covar[t,:,:], [N, 1, 1])
                for i in range(N):
                    mu[i,t,:] = (traj.K[t,:,:].dot(X[i,t,:]) + traj.k[t,:]) - \
                            np.linalg.solve(prc[i,t,:,:],
                            pol_info.lambda_K[t,:,:].dot(X[i,t,:]) + pol_info.lambda_k[t,:])
                wt[:,t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt)

    def _update_policy_fit(self, m, init=False):
        """
        Re-estimate the local policy values in the neighborhood of the trajectory.

        Args:
            m: Condition
            init: Whether this is the initial fitting of the policy.
        """
        dX, dU, T = self.dX, self.dU, self.T
        # Choose samples to use.
        samples = self.cur[m].sample_list
        N = len(samples)
        traj_info, pol_info = self.cur[m].traj_info, self.cur[m].pol_info
        X, obs = samples.get_X(), samples.get_obs()
        pol_mu, pol_sig = self.policy_opt.prob(samples.get_obs())[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig
        # Update policy prior.
        if init:
            #TODO: what arguments need to be passed in here?
            self.policy_prior.update()
        else:
            self.policy_prior.update()
        # Collapse policy covariances.
        # This is not really correct, but it works fine so long as the policy covariance doesn't depend on state.
        pol_sig = np.mean(pol_sig, axis=0)
        # Estimate the policy linearization at each time step.
        for t in range(T):
            # Assemble diagonal weights matrix and data.
            dwts = (1./N) * np.ones(N)
            Ts = X[:,t,:]
            Ps = pol_mu[:,t,:]
            Ys = np.concatenate((Ts, Ps), axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, m, n0 = self.policy_prior.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # On the first time step, always slightly regularize covariance.
            if t == 0:
                sig_reg[:dX,:dX] = 1e-8 * np.eye(dX)
            # Perform computation.
            pol_K, pol_k, pol_S = gauss_fit_joint_prior(Ys, mu0, Phi, m, n0, dwts, dX, dU, sig_reg)
            pol_S += pol_sig[t,:,:]
            pol_info.pol_K[t,:,:], pol_info.pol_k[t,:] = pol_K, pol_k
            pol_info.pol_S[t,:,:], pol_info.chol_pol_S[t,:,:] = pol_S, sp.linalg.cholesky(pol_S)

    def _policy_dual_step(self, m, step=False):
        """
        Update the dual variables for the specified condition.

        Args:
            m: Condition
            step: Whether or not to update pol_wt.
        """
        dX, dU, T = self.dX, self.dU, self.T
        samples = self.cur[m].sample_list
        N = len(samples)
        X, obs = samples.get_X(), samples.get_obs()
        traj, traj_info, pol_info = self.cur[m].traj_distr, self.cur[m].traj_info, self.cur[m].pol_info
        # Compute trajectory action at each sampled state.
        traj_mu = np.zeros((N, T, dU))
        for i in range(N):
            for t in range(T):
                traj_mu[i,t,:] = traj.K[t,:,:].dot(X[i,t,:]) + traj.k[t,:]
        # Compute policy action at each sampled state.
        pol_mu, pol_sig = pol_info.pol_mu, pol_info.pol_sig
        # Compute the first and second moments along the trajectory and the policy.
        traj_ev, traj_em = estimate_moments(X, traj_mu, traj.pol_covar)
        pol_ev, pol_em = estimate_moments(X, pol_mu, pol_sig)
        # Compute the difference and increment based on pol_wt.
        for t in range(T):
            tU, pU = traj_mu[:,t,:], pol_mu[:,t,:]
            # Increment mean term.
            pol_info.lambda_k[t,:] -= self._hyperparams['policy_dual_rate'] * pol_info.pol_wt[t] * \
                    traj.inv_pol_covar[t,:,:].dot(np.mean(tU-pU, axis=0))
            # Increment covariance term.
            t_covar, p_covar = traj.K[t,:,:], pol_info.pol_K[t,:,:]
            pol_info.lambda_K[t,:,:] -= self._hyperparams['policy_dual_rate_covar'] * pol_info.pol_wt[t] * \
                    traj.inv_pol_covar[t,:,:].dot(t_covar - p_covar)
        # Compute KL divergence.
        kl_m = self._policy_kl(m)[0]
        if step:
            # Increment pol_wt based on change in KL divergence.
            if self._hyperparams['fixed_lg_step'] == 1:
                # Take fixed size step.
                pol_info.pol_wt = np.array([max(wt + self._hyperparams['lg_step'], 0) \
                                             for wt in pol_info.pol_wt])
            elif self._hyperparams['fixed_lg_step'] == 2:
                # Increase/decrease based on change in constraint satisfaction.
                if hasattr(traj_info, 'prev_kl'):
                    kl_change = kl_m / pol_info.prev_kl
                    for i in range(len(pol_info.pol_wt)):
                        if kl_change[i] < 0.8:
                            pol_info.pol_wt[i] *= 0.5
                        elif kl_change[i] >= 0.95:
                            pol_info.pol_wt[i] *= 2.0
            elif self._hyperparams['fixed_lg_step'] == 3:
                # Increase/decrease based on difference from average.
                if hasattr(traj_info, 'prev_kl'):
                    lower = np.mean(kl_m) - self._hyperparams['exp_step_lower'] * np.std(kl_m)
                    upper = np.mean(kl_m) + self._hyperparams['exp_step_upper'] * np.std(kl_m)
                    for i in range(len(pol_info.pol_wt)):
                        if kl_m[i] < lower:
                            pol_info.pol_wt[i] *= self._hyperparams['exp_step_decrease']
                        elif kl_m[i] >= upper:
                            pol_info.pol_wt[i] *= self._hyperparams['exp_step_increase']
            else:
                # Standard DGD step.
                pol_info.pol_wt = np.array([max(pol_info.pol_wt[i] + self._hyperparams['lg_step']*kl_m[i], 0) \
                                             for i in range(T)])
            pol_info.prev_kl = kl_m
        self.cur[m].pol_kl = kl_m

    def _update_trajectories(self):
        """
        Compute new linear gaussian controllers.
        """
        if not hasattr(self, 'new_traj_distr'):
            self.new_traj_distr = [self.cur[m].traj_distr for m in range(self.M)]
        for m in range(self.M):
            self.new_traj_distr[m], self.eta[m] = self.traj_opt.update(
                    self.T, self.cur[m].step_mult, self.eta[m],
                    self.cur[m].traj_info, self.new_traj_distr[m])

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
            self.cur[m].traj_info.last_kl_step = self.prev[m].traj_info.last_kl_step
            pol_info, prev_pol_info = PolicyInfo(), self.prev[m].pol_info
            pol_info.lambda_k = np.copy(prev_pol_info.lambda_k)
            pol_info.lambda_K = np.copy(prev_pol_info.lambda_K)
            pol_info.pol_wt = np.copy(prev_pol_info.pol_wt)
            pol_info.pol_K = np.copy(prev_pol_info.pol_K)
            pol_info.pol_k = np.copy(prev_pol_info.pol_k)
            pol_info.pol_S = np.copy(prev_pol_info.pol_S)
            pol_info.chol_pol_S = np.copy(prev_pol_info.chol_pol_S)
            self.cur[m].pol_info = pol_info
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].traj_distr = self.new_traj_distr[m]
        delattr(self, 'new_traj_distr')

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

    def _stepadjust(self, m):
        """
        Calculate new step sizes.

        Args:
            m: Condition
        """
        # Compute values under Laplace approximation.
        # This is the policy that the previous samples were actually drawn from
        # under the dynamics that were estimated from the previous samples.
        previous_laplace_obj, previous_laplace_kl = self._estimate_cost(self.prev[m].traj_distr, self.prev[m].traj_info, m)
        # This is the policy that we just used under the dynamics that were
        # estimated from the previous samples (so this is the cost we thought we
        # would have).
        new_predicted_laplace_obj, new_predicted_laplace_kl = self._estimate_cost(self.cur[m].traj_distr, self.prev[m].traj_info, m)

        # This is the actual cost we have under the current trajectory based on the
        # latest samples.
        new_actual_laplace_obj, new_actual_laplace_kl = self._estimate_cost(self.cur[m].traj_distr, self.cur[m].traj_info, m)

        # Measure the entropy of the current trajectory (for printout).
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t, :, :])))

        # Compute actual objective values based on the samples.
        previous_mc_obj = np.mean(np.sum(self.prev[m].cs, axis=1), axis=0)
        new_mc_obj = np.mean(np.sum(self.cur[m].cs, axis=1), axis=0)

        # Compute sample-based estimate of KL divergence between policy and trajectories.
        new_mc_kl, new_mc_kl_samp, new_mc_lam, new_mc_lam_samp = self._policy_kl(m)
        if self.iteration_count >= 1 and self.prev[m].sample_list:
            previous_mc_kl, previous_mk_kl_samp, previous_mc_lam, previous_mc_lam_samp = \
                    self._policy_kl(m, prev=True)
        else:
            previous_mc_kl = np.zeros_like(new_mc_kl)

        # Compute full policy KL divergence objective terms by applying the Lagrange multipliers.
        previous_laplace_kl_sum = np.sum(previous_laplace_kl * self.cur[m].pol_info.pol_wt)
        new_predicted_laplace_kl_sum = np.sum(new_predicted_laplace_kl * self.cur[m].pol_info.pol_wt)
        new_actual_laplace_kl_sum = np.sum(new_actual_laplace_kl * self.cur[m].pol_info.pol_wt)
        previous_mc_kl_sum = np.sum(previous_mc_kl * self.cur[m].pol_info.pol_wt)
        new_mc_kl_sum = np.sum(new_mc_kl * self.cur[m].pol_info.pol_wt)
        new_mc_lam_sum = np.sum(new_mc_lam * self.cur[m].pol_info.pol_wt)

        # Compute misprediction vs Monte-Carlo score.
        mispred_std = np.abs(np.sum(new_actual_laplace_obj) + new_actual_laplace_kl_sum - new_mc_obj - new_mc_lam_sum) / \
                max(np.std(np.sum(self.cur[m].cs + new_mc_lam_samp * self.cur[m].pol_info.pol_wt, axis=1), axis=0), 1.0)

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f KL: %f -> %f', ent, previous_mc_obj, new_mc_obj,
                previous_mc_kl_sum, new_mc_kl_sum)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) + previous_laplace_kl_sum - np.sum(new_predicted_laplace_obj) - new_predicted_laplace_kl_sum
        actual_impr = np.sum(previous_laplace_obj) + previous_laplace_kl_sum - np.sum(new_actual_laplace_obj) - new_actual_laplace_kl_sum

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f', np.sum(previous_laplace_obj), previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f', np.sum(new_predicted_laplace_obj), new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f', np.sum(new_actual_laplace_obj), new_mc_obj)
        LOGGER.debug('Previous KL: Laplace: %f MC: %f', np.sum(previous_laplace_kl), np.sum(previous_mc_kl))
        LOGGER.debug('Predicted new KL: Laplace: %f MC: %f', np.sum(new_predicted_laplace_kl), np.sum(new_mc_kl))
        LOGGER.debug('Actual new KL: Laplace: %f MC: %f', np.sum(new_actual_laplace_kl), np.sum(new_mc_kl))
        LOGGER.debug('Previous w KL: Laplace: %f MC: %f', previous_laplace_kl_sum, previous_mc_kl_sum)
        LOGGER.debug('Predicted w new KL: Laplace: %f MC: %f', new_predicted_laplace_kl_sum, new_mc_kl_sum)
        LOGGER.debug('Actual w new KL: Laplace %f MC: %f', new_actual_laplace_kl_sum, new_mc_kl_sum)
        LOGGER.debug('Predicted/actual improvement: %f / %f', predicted_impr, actual_impr)

        # Compute actual KL step taken at last iteration.
        actual_step = self.cur[m].traj_info.last_kl_step / (self._hyperparams['kl_step'] * self.T)
        if actual_step < self.cur[m].step_mult:
            self.cur[m].step_mult = max(actual_step, self._hyperparams['min_step_mult'])

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
        self.cur[m].pol_kl = new_mc_kl

    def _policy_kl(self, m, prev=False):
        """
        Monte-Carlo estimate of KL divergence between policy and trajectory.
        """
        dX, dU, T = self.dX, self.dU, self.T
        samples = self.cur[m].sample_list
        N = len(samples)
        X, obs = samples.get_X(), samples.get_obs()
        if prev:
            traj, pol_info, traj_info = self.prev[m].traj_distr, self.cur[m].pol_info, self.prev[m].traj_info
            samples = self.prev[m].sample_list
            N = len(samples)
        else:
            traj, pol_info, traj_info = self.cur[m].traj_distr, self.cur[m].pol_info, self.cur[m].traj_info
            samples = self.cur[m].sample_list
            N = len(samples)
        kl, kl_m = np.zeros((N,T)), np.zeros(T)
        kl_l, kl_lm = np.zeros((N,T)), np.zeros(T)
        # Compute policy mean and covariance at each sample.
        pol_mu, _, pol_prec, pol_det_sigma = self.policy_opt.prob(samples.get_obs())
        # Compute KL divergence.
        for t in range(T):
            # Compute trajectory action at sample.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i,:] = traj.K[t,:,:].dot(X[i,t,:]) + traj.k[t,:]
            # Compute KL divergence.
            diff = pol_mu[:,t,:] - traj_mu
            #TODO: rename these
            term1 = pol_prec[:,t,:,:] * traj.pol_covar[t,:,:]
            term2 = 0.5 * dU + np.sum(np.log(np.diag(traj.chol_pol_covar[t,:,:])))
            term3 = np.log(pol_det_sigma[:,t])
            # IMPORTANT: note that this assumes that pol_prec does not depend
            # on state!!!! (only the last term makes this assumption)
            term4 = np.sum(diff * (diff.dot(pol_prec[1,t,:,:])), axis=1)
            kl[:,t] = 0.5 * np.sum(np.sum(term1, axis=1), axis=1) - term2 + \
                    0.5 * term3 + 0.5 * term4
            kl_m[t] = 0.5 * np.sum(np.sum(np.mean(term1, axis=0), axis=0), axis=0) - term2 + \
                    0.5 * np.mean(term3) + 0.5 * np.mean(term4)
            # Compute trajectory action at sample with Lagrange multiplier.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i,:] = (traj.K[t,:,:] - pol_info.lambda_K[t,:,:]).dot(X[i,t,:]) + \
                        (traj.k[t,:] - pol_info.lambda_k[t,:])
            # Compute KL divergence with Lagrange multiplier.
            diff_l = pol_mu[:,t,:] - traj_mu
            term4_l = np.sum(diff_l * (diff_l.dot(pol_prec[1,t,:,:])), axis=1)
            kl_l[:,t] = 0.5 * np.sum(np.sum(term1, axis=1), axis=1) - term2 + \
                    0.5 * term3 + 0.5 * term4_l
            kl_lm[t] = 0.5 * np.sum(np.sum(np.mean(term1, axis=0), axis=0), axis=0) - term2 + \
                    0.5 * np.mean(term3) + 0.5 * np.mean(term4_l)
        return kl_m, kl, kl_lm, kl_l

    def _estimate_cost(self, traj_distr, traj_info, m):
        """
        Compute Laplace approximation to expected cost.

        Args:
            traj_distr: Linear gaussian policy object
            traj_info:
        """
        pol_info = self.cur[m].pol_info

        # Constants.
        T = self.T
        dU = self.dU
        dX = self.dX

        # Perform forward pass (note that we repeat this here, because traj_info may
        # have different dynamics from the ones that were used to compute the
        # distribution already saved in traj).
        mu, sigma = self.traj_opt.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + 0.5 * np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + \
                0.5 * mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + mu[t, :].T.dot(traj_info.cv[t, :])

        # Compute KL divergence.
        predicted_kl = np.zeros(T)
        for t in range(T):
            inv_pS = np.linalg.solve(pol_info.chol_pol_S[t,:,:],
                    np.linalg.solve(pol_info.chol_pol_S[t,:,:].T, np.eye(dU)))
            Ufb = pol_info.pol_K[t,:,:].dot(mu[t,:dX].T) + pol_info.pol_k[t,:]
            Kbar = traj_distr.K[t,:,:] - pol_info.pol_K[t,:,:]
            predicted_kl[t] = 0.5 * (mu[t,dX:] - Ufb).dot(inv_pS).dot(mu[t,dX:] - Ufb) + \
                    0.5 * np.sum(traj_distr.pol_covar[t,:,:] * inv_pS) + \
                    0.5 * np.sum(sigma[t,:dX,:dX] * Kbar.T.dot(inv_pS).dot(Kbar)) + \
                    np.sum(np.log(np.diag(pol_info.chol_pol_S[t,:,:]))) - \
                    (np.sum(np.log(np.diag(traj_distr.chol_pol_covar[t,:,:]))) - 0.5 * dU)

        return predicted_cost, predicted_kl
