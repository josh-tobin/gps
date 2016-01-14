import copy
import numpy as np
import scipy as sp
import logging

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.algorithm_utils import estimate_moments, gauss_fit_joint_prior, IterationData, \
        TrajectoryInfo, PolicyInfo
from gps.algorithm.config import alg_badmm
from gps.sample.sample_list import SampleList
from gps.utility.general_utils import extract_condition


LOGGER = logging.getLogger(__name__)


class AlgorithmBADMM(Algorithm):
    """
    Sample-based joint policy learning and trajectory optimization with BADMM.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(alg_badmm)
        config.update(hyperparams)
        Algorithm.__init__(self, config)

        # IterationData objects for each condition.
        self.cur = [IterationData() for _ in range(self.M)]
        self.prev = [IterationData() for _ in range(self.M)]

        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            dynamics = self._hyperparams['dynamics']
            self.cur[m].traj_info.dynamics = dynamics['type'](dynamics)
            init_traj_distr = extract_condition(self._hyperparams['init_traj_distr'], m)
            self.cur[m].traj_distr = init_traj_distr['type'](init_traj_distr)
            self.cur[m].pol_info = PolicyInfo(self._hyperparams)
            policy_prior = self._hyperparams['policy_prior']
            self.cur[m].pol_info.policy_prior = policy_prior['type'](policy_prior)

        self.policy_opt = self._hyperparams['policy_opt']['type'](self._hyperparams['policy_opt'],
                self.dO, self.dU)

    def iteration(self, sample_lists):
        """
        Run iteration of BADMM-based guided policy search.
        Args:
            sample_lists: List of sample_list objects for each condition.
        """
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]

        self._set_interp_values()
        self._update_dynamics()  # Update dynamics model using all sample.
        self._update_policy_samples()  # Choose the samples to use with the policy.
        self._update_step_size()  # KL Divergence step size.

        # Run inner loop to compute new policies under new dynamics and step size.
        for inner_itr in range(self._hyperparams['inner_iterations']):
            #TODO: Could start from init controller.
            if self.iteration_count > 0 or inner_itr > 0:
                self._update_policy(self.iteration_count, inner_itr)  # Update the policy.
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
        t = min((self.iteration_count+1.0) / (self._hyperparams['iterations']-1), 1)
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

    def _update_policy_samples(self):
        #TODO: Handle synthetic samples.
        if self._hyperparams['policy_sample_mode'] == 'add':
            for m in range(self.M):
                samples = self.cur[m].pol_info.policy_samples
                samples.extend(self.cur[m].sample_list)
                if len(samples) > self._hyperparams['max_policy_samples']:
                    start = len(samples) - self._hyperparams['max_policy_samples']
                    self.cur[m].pol_info.policy_samples = samples[start:]
        else:
            for m in range(self.M):
                self.cur[m].pol_info.policy_samples = self.cur[m].sample_list

    def _update_step_size(self):
        """
        Evaluate costs on samples, and adjust the step size.
        """
        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._update_policy_fit(m, init=True)
            self._eval_cost(m)
            # Adjust step size relative to the previous iteration.
            if self.iteration_count >= 1 and self.prev[m].sample_list:
                self._stepadjust(m)

    def _update_policy(self, itr, inner_itr):
        """
        Compute the new policy.
        """
        dX, dU, dO, T = self.dX, self.dU, self.dO, self.T
        # Compute target mean, cov, and weight for each sample.
        tgt_mu, tgt_prc, tgt_wt = np.zeros((0, T, dU)), np.zeros((0, T, dU, dU)), np.zeros((0, T))
        obs_data = np.zeros((0, T, dO))
        for m in range(self.M):
            samples = self.cur[m].sample_list
            X, U = samples.get_X(), samples.get_U()
            N = len(samples)
            traj, traj_info = self.cur[m].traj_distr, self.cur[m].traj_info
            pol_info = self.cur[m].pol_info
            mu = np.zeros((N, T, dU))
            prc = np.zeros((N, T, dU, dU))
            wt = np.zeros((N, T))
            # Get time-indexed actions.
            for t in range(T):
                # Compute actions along this trajectory.
                prc[:,t,:,:] = np.tile(traj.inv_pol_covar[t,:,:], [N, 1, 1])
                for i in range(N):
                    mu[i,t,:] = (traj.K[t,:,:].dot(X[i,t,:]) + traj.k[t,:]) - \
                            np.linalg.solve(prc[i,t,:,:],  #TODO: Divide by pol_wt[t].
                            pol_info.lambda_K[t,:,:].dot(X[i,t,:]) + pol_info.lambda_k[t,:])
                wt[:,t].fill(pol_info.pol_wt[t])
            tgt_mu = np.concatenate((tgt_mu, mu))
            tgt_prc = np.concatenate((tgt_prc, prc))
            tgt_wt = np.concatenate((tgt_wt, wt))
            obs_data = np.concatenate((obs_data, samples.get_obs()))
        self.policy_opt.update(obs_data, tgt_mu, tgt_prc, tgt_wt, itr, inner_itr)

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
        X = samples.get_X()
        pol_mu, pol_sig = self.policy_opt.prob(samples.get_obs().copy())[:2]
        pol_info.pol_mu, pol_info.pol_sig = pol_mu, pol_sig
        # Update policy prior.
        if init:
            self.cur[m].pol_info.policy_prior.update(samples, self.policy_opt,
                    SampleList(self.cur[m].pol_info.policy_samples))
        else:
            self.cur[m].pol_info.policy_prior.update(SampleList([]), self.policy_opt,
                    SampleList(self.cur[m].pol_info.policy_samples))
        # Collapse policy covariances. This is not really correct, but it works fine so long as the
        # policy covariance doesn't depend on state.
        pol_sig = np.mean(pol_sig, axis=0)
        # Estimate the policy linearization at each time step.
        for t in range(T):
            # Assemble diagonal weights matrix and data.
            dwts = (1./N) * np.ones(N)
            Ts = X[:,t,:]
            Ps = pol_mu[:,t,:]
            Ys = np.concatenate((Ts, Ps), axis=1)
            # Obtain Normal-inverse-Wishart prior.
            mu0, Phi, mm, n0 = self.cur[m].pol_info.policy_prior.eval(Ts, Ps)
            sig_reg = np.zeros((dX+dU, dX+dU))
            # On the first time step, always slightly regularize covariance.
            if t == 0:
                sig_reg[:dX,:dX] = 1e-8 * np.eye(dX)
            # Perform computation.
            pol_K, pol_k, pol_S = gauss_fit_joint_prior(Ys, mu0, Phi, mm, n0, dwts, dX, dU, sig_reg)
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
        traj, traj_info = self.cur[m].traj_distr, self.cur[m].traj_info
        pol_info = self.cur[m].pol_info
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
            pol_info.lambda_K[t,:,:] -= self._hyperparams['policy_dual_rate_covar'] * \
                    pol_info.pol_wt[t] * traj.inv_pol_covar[t,:,:].dot(t_covar - p_covar)
        # Compute KL divergence.
        kl_m = self._policy_kl(m)[0]
        if step:
            # Increment pol_wt based on change in KL divergence.
            if self._hyperparams['fixed_lg_step'] == 1:
                # Take fixed size step.
                pol_info.pol_wt = np.array(
                        [max(wt + self._hyperparams['lg_step'], 0) for wt in pol_info.pol_wt])
            elif self._hyperparams['fixed_lg_step'] == 2:
                # Increase/decrease based on change in constraint satisfaction.
                if hasattr(pol_info, 'prev_kl'):
                    kl_change = kl_m / pol_info.prev_kl
                    for i in range(len(pol_info.pol_wt)):
                        if kl_change[i] < 0.8:
                            pol_info.pol_wt[i] *= 0.5
                        elif kl_change[i] >= 0.95:
                            pol_info.pol_wt[i] *= 2.0
            elif self._hyperparams['fixed_lg_step'] == 3:
                # Increase/decrease based on difference from average.
                if hasattr(pol_info, 'prev_kl'):
                    lower = np.mean(kl_m) - self._hyperparams['exp_step_lower'] * np.std(kl_m)
                    upper = np.mean(kl_m) + self._hyperparams['exp_step_upper'] * np.std(kl_m)
                    for i in range(len(pol_info.pol_wt)):
                        if kl_m[i] < lower:
                            pol_info.pol_wt[i] *= self._hyperparams['exp_step_decrease']
                        elif kl_m[i] >= upper:
                            pol_info.pol_wt[i] *= self._hyperparams['exp_step_increase']
            else:
                # Standard DGD step.
                pol_info.pol_wt = np.array(
                        [max(pol_info.pol_wt[i] + self._hyperparams['lg_step']*kl_m[i], 0)
                                for i in range(T)])
            pol_info.prev_kl = kl_m

    def _advance_iteration_variables(self):
        """
        Move all 'cur' variables to 'prev', and advance iteration counter.
        """
        self.iteration_count += 1
        self.prev = self.cur
        self.cur = [IterationData() for _ in range(self.M)]
        for m in range(self.M):
            self.cur[m].traj_info = TrajectoryInfo()
            self.cur[m].traj_info.last_kl_step = self.prev[m].traj_info.last_kl_step
            self.cur[m].traj_info.dynamics = self.prev[m].traj_info.dynamics
            self.cur[m].step_mult = self.prev[m].step_mult
            self.cur[m].eta = self.prev[m].eta
            self.cur[m].traj_distr = self.new_traj_distr[m]
            self.cur[m].pol_info = self.prev[m].pol_info
        delattr(self, 'new_traj_distr')

    def _stepadjust(self, m):
        """
        Calculate new step sizes.
        Args:
            m: Condition
        """
        # Compute values under Laplace approximation. This is the policy that the previous samples
        # were actually drawn from under the dynamics that were estimated from the previous samples.
        previous_laplace_obj, previous_laplace_kl = self._estimate_cost(self.prev[m].traj_distr,
                self.prev[m].traj_info, m)
        # This is the policy that we just used under the dynamics that were estimated from the
        # previous samples (so this is the cost we thought we would have.
        new_predicted_laplace_obj, new_predicted_laplace_kl = self._estimate_cost(
                self.cur[m].traj_distr, self.prev[m].traj_info, m)

        # This is the actual cost we have under the current trajectory based on the latest samples.
        new_actual_laplace_obj, new_actual_laplace_kl = self._estimate_cost(self.cur[m].traj_distr,
                self.cur[m].traj_info, m)

        # Measure the entropy of the current trajectory (for printout).
        ent = 0
        for t in range(self.T):
            ent = ent + np.sum(np.log(np.diag(self.cur[m].traj_distr.chol_pol_covar[t,:,:])))

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

        LOGGER.debug('Trajectory step: ent: %f cost: %f -> %f KL: %f -> %f', ent, previous_mc_obj,
                new_mc_obj, previous_mc_kl_sum, new_mc_kl_sum)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) + previous_laplace_kl_sum - \
                np.sum(new_predicted_laplace_obj) - new_predicted_laplace_kl_sum
        actual_impr = np.sum(previous_laplace_obj) + previous_laplace_kl_sum - \
                np.sum(new_actual_laplace_obj) - new_actual_laplace_kl_sum

        # Print improvement details.
        LOGGER.debug('Previous cost: Laplace: %f MC: %f', np.sum(previous_laplace_obj),
                previous_mc_obj)
        LOGGER.debug('Predicted new cost: Laplace: %f MC: %f', np.sum(new_predicted_laplace_obj),
                new_mc_obj)
        LOGGER.debug('Actual new cost: Laplace: %f MC: %f', np.sum(new_actual_laplace_obj),
                new_mc_obj)
        LOGGER.debug('Previous KL: Laplace: %f MC: %f', np.sum(previous_laplace_kl),
                np.sum(previous_mc_kl))
        LOGGER.debug('Predicted new KL: Laplace: %f MC: %f', np.sum(new_predicted_laplace_kl),
                np.sum(new_mc_kl))
        LOGGER.debug('Actual new KL: Laplace: %f MC: %f', np.sum(new_actual_laplace_kl),
                np.sum(new_mc_kl))
        LOGGER.debug('Previous w KL: Laplace: %f MC: %f', previous_laplace_kl_sum,
                previous_mc_kl_sum)
        LOGGER.debug('Predicted w new KL: Laplace: %f MC: %f', new_predicted_laplace_kl_sum,
                new_mc_kl_sum)
        LOGGER.debug('Actual w new KL: Laplace %f MC: %f', new_actual_laplace_kl_sum, new_mc_kl_sum)
        LOGGER.debug('Predicted/actual improvement: %f / %f', predicted_impr, actual_impr)

        # Compute actual KL step taken at last iteration.
        actual_step = self.cur[m].traj_info.last_kl_step / (self._hyperparams['kl_step'] * self.T)
        if actual_step < self.cur[m].step_mult:
            self.cur[m].step_mult = max(actual_step, self._hyperparams['min_step_mult'])

        # Model improvement as I = predicted_dI * KL + penalty * KL^2, where predicted_dI = pred/KL
        # and penalty = (act-pred)/(KL^2).
        # Optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL
        # => KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL.
        # Therefore, the new multiplier is given by pred/2*(pred-act).
        new_mult = predicted_impr / (2.0 * max(1e-4, predicted_impr - actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.cur[m].step_mult, self._hyperparams['max_step_mult']),
                self._hyperparams['min_step_mult'])
        self.cur[m].step_mult = new_step

        if new_mult > 1:
            LOGGER.debug('Increasing step size multiplier to %f', new_step)
        else:
            LOGGER.debug('Decreasing step size multiplier to %f', new_step)

    def _policy_kl(self, m, prev=False):
        """
        Monte-Carlo estimate of KL divergence between policy and trajectory.
        """
        dX, dU, T = self.dX, self.dU, self.T
        samples = self.cur[m].sample_list
        N = len(samples)
        X, obs = samples.get_X(), samples.get_obs()
        if prev:
            traj, traj_info = self.prev[m].traj_distr, self.prev[m].traj_info
            pol_info = self.cur[m].pol_info
            samples = self.prev[m].sample_list
            N = len(samples)
        else:
            traj, traj_info = self.cur[m].traj_distr, self.cur[m].traj_info
            pol_info = self.cur[m].pol_info
            samples = self.cur[m].sample_list
            N = len(samples)
        kl, kl_m = np.zeros((N,T)), np.zeros(T)
        kl_l, kl_lm = np.zeros((N,T)), np.zeros(T)
        # Compute policy mean and covariance at each sample.
        pol_mu, _, pol_prec, pol_det_sigma = self.policy_opt.prob(samples.get_obs().copy())
        # Compute KL divergence.
        for t in range(T):
            # Compute trajectory action at sample.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i,:] = traj.K[t,:,:].dot(X[i,t,:]) + traj.k[t,:]
            diff = pol_mu[:,t,:] - traj_mu
            tr_pp_ct = pol_prec[:,t,:,:] * traj.pol_covar[t,:,:]
            k_ln_det_ct = 0.5 * dU + np.sum(np.log(np.diag(traj.chol_pol_covar[t,:,:])))
            ln_det_cp = np.log(pol_det_sigma[:,t])
            # IMPORTANT: note that this assumes that pol_prec does not depend on state!!!! (only the
            #            last term makes this assumption)
            d_pp_d = np.sum(diff * (diff.dot(pol_prec[1,t,:,:])), axis=1)
            kl[:,t] = 0.5 * np.sum(np.sum(tr_pp_ct, axis=1), axis=1) - k_ln_det_ct + \
                    0.5 * ln_det_cp + 0.5 * d_pp_d
            kl_m[t] = 0.5 * np.sum(np.sum(np.mean(tr_pp_ct, axis=0), axis=0), axis=0) - k_ln_det_ct \
                    + 0.5 * np.mean(ln_det_cp) + 0.5 * np.mean(d_pp_d)
            # Compute trajectory action at sample with Lagrange multiplier.
            traj_mu = np.zeros((N, dU))
            for i in range(N):
                traj_mu[i,:] = (traj.K[t,:,:] - pol_info.lambda_K[t,:,:]).dot(X[i,t,:]) + \
                        (traj.k[t,:] - pol_info.lambda_k[t,:])
            # Compute KL divergence with Lagrange multiplier.
            diff_l = pol_mu[:,t,:] - traj_mu
            d_pp_d_l = np.sum(diff_l * (diff_l.dot(pol_prec[1,t,:,:])), axis=1)
            kl_l[:,t] = 0.5 * np.sum(np.sum(tr_pp_ct, axis=1), axis=1) - k_ln_det_ct + \
                    0.5 * ln_det_cp + 0.5 * d_pp_d_l
            kl_lm[t] = 0.5 * np.sum(np.sum(np.mean(tr_pp_ct, axis=0), axis=0), axis=0) - \
                    k_ln_det_ct + 0.5 * np.mean(ln_det_cp) + 0.5 * np.mean(d_pp_d_l)
        return kl_m, kl, kl_lm, kl_l

    def _estimate_cost(self, traj_distr, traj_info, m):
        """
        Compute Laplace approximation to expected cost.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
            m: Condition number.
        """
        pol_info = self.cur[m].pol_info

        # Constants.
        T = self.T
        dU = self.dU
        dX = self.dX

        # Perform forward pass (note that we repeat this here, because traj_info may have different
        # dynamics from the ones that were used to compute the distribution already saved in traj).
        mu, sigma = self.traj_opt.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + \
                    0.5 * np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + \
                    0.5 * mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + \
                    mu[t, :].T.dot(traj_info.cv[t, :])

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

    def _compute_costs(self, m, eta):
        """
        Compute cost estimates used in the LQR backward pass.
        """
        traj_info, traj_distr = self.cur[m].traj_info, self.cur[m].traj_distr
        pol_info = self.cur[m].pol_info
        T, dU, dX = traj_distr.T, traj_distr.dU, traj_distr.dX
        Cm, cv = np.copy(traj_info.Cm), np.copy(traj_info.cv)

        # Modify policy action via Lagrange multiplier.
        cv[:,dX:] -= pol_info.lambda_k
        Cm[:,dX:,:dX] -= pol_info.lambda_K
        Cm[:,:dX,dX:] -= np.transpose(pol_info.lambda_K, [0, 2, 1]) 

        #Pre-process the costs with KL-divergence terms.
        TKLm = np.zeros((T,dX+dU,dX+dU))
        TKLv = np.zeros((T,dX+dU))
        PKLm = np.zeros((T,dX+dU,dX+dU))
        PKLv = np.zeros((T,dX+dU))
        fCm, fcv = np.zeros(Cm.shape), np.zeros(cv.shape)
        for t in range(T):
            K, k = traj_distr.K[t,:,:], traj_distr.k[t,:]
            inv_pol_covar = traj_distr.inv_pol_covar[t,:,:]
            # Trajectory KL-divergence terms.
            TKLm[t,:,:] = np.vstack([
                np.hstack([K.T.dot(inv_pol_covar).dot(K), -K.T.dot(inv_pol_covar)]),
                np.hstack([-inv_pol_covar.dot(K), inv_pol_covar])])
            TKLv[t,:] = np.concatenate([K.T.dot(inv_pol_covar).dot(k), -inv_pol_covar.dot(k)])
            # Policy KL-divergence terms.
            inv_pol_S = np.linalg.solve(pol_info.chol_pol_S[t,:,:],
                np.linalg.solve(pol_info.chol_pol_S[t,:,:].T, np.eye(dU)))
            KB, kB = pol_info.pol_K[t,:,:], pol_info.pol_k[t,:]
            PKLm[t,:,:] = np.vstack([
                np.hstack([KB.T.dot(inv_pol_S).dot(KB), -KB.T.dot(inv_pol_S)]),
                np.hstack([-inv_pol_S.dot(KB), inv_pol_S])])
            PKLv[t,:] = np.concatenate([KB.T.dot(inv_pol_S).dot(kB), -inv_pol_S.dot(kB)])
            wt = pol_info.pol_wt[t]
            fCm[t,:,:] = (Cm[t,:,:] + TKLm[t,:,:] * eta + PKLm[t,:,:] * wt) / (eta + wt)
            fcv[t,:] = (cv[t,:] + TKLv[t,:] * eta + PKLv[t,:] * wt) / (eta + wt)

        return fCm, fcv
