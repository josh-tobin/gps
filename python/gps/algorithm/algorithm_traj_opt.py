from algorithm import Algorithm
import numpy as np


class AlgorithmTrajOpt(Algorithm):
    """Sample-based trajectory optimization.

    """

    def __init__(self, hyperparams, sample_data):
        Algorithm.__init__(self, hyperparams, sample_data)
        self.traj_opt = hyperparams['traj_opt']['type'](hyperparams['traj_opt'], sample_data, self.dynamics)
        self.costs = None  # TODO: Init costs from hyperparams
        # TODO - Initialize trajectory distributions from hyperparams
        self.max_step_mult = hyperparams['max_step_mult']
        self.min_step_mult = hyperparams['min_step_mult']
        self.M = 0

        self.iteration = 0  # Keep track of what iteration this is currently on
        # TODO: Remove. This is very hacky
        # List of variables updated from iteration to iteration
        self.iteration_vars = ['samples', 'refs', 'trajinfo_cc', 'trajinfo_cv', 'trajinfo_Cm', 'traj_distr', 'cs',
                               'rate_change', 'mispred_std', 'init_polkl', 'traj_distr_step_mult']

        # TODO: Remove. This is very hacky
        for varname in self.iteration_vars:
            setattr(self, 'cur_' + varname, None)
            setattr(self, 'prev_' + varname, None)

    def iteration(self):
        """
        Run iteration of LQR.
        """
        self.advance_timestep()

        # Update dynamics model using all sample.
        self.dynamics.update_prior()
        self.dynamics.fit()

        self.eval_costs()
        self.update_step()

        # Run inner loop
        for inner_itr in range(self._hyperparams['inner_iterations']):
            self.traj_opt.update()

    def update_step(self):
        """ Evaluate costs on samples, adjusts step size """
        # Evaluate cost function.
        for m in range(self.M):  # m = condition
            if self.iteration > 1 and self.prev_samples[m]:
                # Evaluate cost and adjust step size relative to the previous iteration.
                self.stepadjust(m)

    def stepadjust(self, m):
        """
        Calculate new step sizes.

        This code does not run yet.
        """
        T = 0
        # No policy by default.
        polkl = np.zeros(T)

        # Compute values under Laplace approximation.
        # This is the policy that the previous samples were actually drawn from
        # under the dynamics that were estimated from the previous samples.
        previous_laplace_obj = self.traj_opt.estimate_cost(self.prev_traj_distr[m],
                                                           self.prev_trajinfo_cc[m],
                                                           self.prev_trajinfo_cv[m],
                                                           self.prev_trajinfo_Cm[m])

        # This is the policy that we just used under the dynamics that were
        # estimated from the previous samples (so this is the cost we thought we
        # would have).
        new_predicted_laplace_obj = self.traj_opt.estimate_cost(self.cur_traj_distr[m],
                                                                self.prev_trajinfo_cc[m],
                                                                self.prev_trajinfo_cv[m],
                                                                self.prev_trajinfo_Cm[m])

        # This is the actual cost we have under the current trajectory based on the
        # latest samples.
        new_actual_laplace_obj = self.traj_opt.estimate_cost(self.cur_traj_distr[m],
                                                             self.cur_trajinfo_cc[m],
                                                             self.cur_trajinfo_cv[m],
                                                             self.cur_trajinfo_Cm[m])

        # Measure the entropy of the current trajectory (for printout).
        ent = 0
        for t in range(T):
            ent = ent + np.sum(np.log(np.diag(self.cur_traj_distr[m].cholPSig[:, :, t])))

        # Compute actual objective values based on the samples.
        # previous_mc_obj = np.mean(np.sum(prev_itr_data.ics, axis=1), axis=2)
        new_mc_obj = np.mean(np.sum(self.cur_cs[m], axis=0), axis=1)

        # Compute misprediction vs Monte-Carlo score.
        mispred_std = np.abs(np.sum(new_actual_laplace_obj) - new_mc_obj) / np.max(
            np.std(np.sum(self.cur_cs[m], axis=0), axis=1), 1.0)

        # Compute predicted and actual improvement.
        predicted_impr = np.sum(previous_laplace_obj) - np.sum(new_predicted_laplace_obj)
        actual_impr = np.sum(previous_laplace_obj) - np.sum(new_actual_laplace_obj)

        # model improvement as: I = predicted_dI * KL + penalty * KL^2
        # where predicted_dI = pred/KL and penalty = (act-pred)/(KL^2)
        # optimize I w.r.t. KL: 0 = predicted_dI + 2 * penalty * KL => KL' = (-predicted_dI)/(2*penalty) = (pred/2*(pred-act)) * KL
        # therefore, the new multiplier is given by pred/2*(pred-act)
        new_mult = predicted_impr / (2 * np.max(1e-4, predicted_impr - actual_impr))
        new_mult = np.max(0.1, np.min(5.0, new_mult))
        new_step = np.max(np.min(new_mult * self.cur_traj_distr_step_mult[m], self.max_step_mult), self.min_step_mult)
        step_change = new_step / self.cur_traj_distr_step_mult[m]
        self.cur_traj_distr_step_mult[m] = new_step

        self.cur_step_change[m] = step_change
        self.cur_mispred_std[m] = mispred_std
        self.cur_polkl[m] = polkl

    def eval_costs(self):
        """
        Evaluate costs for all conditions and samples.
        """
        for m in range(self.M):
            self.eval_cost(m)

    # TODO: Update code so that it runs. Clean args and names of variables
    def eval_cost(self, m):
        """
        Evaluate costs for all samples for a condition
        This code does not run yet.
        """
        samples = self.cur_samples[m]
        refs = self.refs[m]
        # Constants.
        Dx = 0
        Du = 0
        T = 0
        N = len(samples)

        # Compute cost.
        cs = np.zeros((N, T))
        cc = np.zeros((N, T))
        cv = np.zeros((N, T, Dx + Du))
        Cm = np.zeros((N, T, Dx + Du, Dx + Du))
        for n in range(N):
            sample = samples[n]
            # Get costs.
            l, lx, lu, lxx, luu, lux = self.costs[m].eval(sample)

            #TODO: Transposes are wrong
            cc[n, :] = l
            cs[n, :] = l
            # Assemble matrix and vector.
            cv[n, :, :] = np.vstack([lx, lu])
            Cm[n, :, :, :] = np.c_[np.r_[lxx, np.transpose(lux, [1, 0, 2])],
                                   np.r_[lux, luu]]
            # Adjust for difference from reference.
            yhat = np.c_[sample.get_X()[n, :, :], sample.get_U()[n, :, :]]

            rdiff = refs[n][:, 1:T] - yhat
            cc[n, :] = cc[n, :] + np.sum(rdiff * cv[n, :, :], axis=0) + \
                       0.5 * np.sum(
                           rdiff * np.transpose(np.sum(Cm[n, :, :, :] * np.transpose(rdiff, [2, 0, 1]), axis=1),
                                                [0, 2, 1]), axis=0)
            cv[n, :, :] = cv[n, :, :] + np.transpose(np.sum(Cm[n, :, :, :] * np.transpose(rdiff, [2, 0, 1]), axis=1),
                                                     [0, 2, 1])
        self.cur_trajinfo_cc(np.mean(cc, 0))  # Costs
        self.cur_trajinfo_cv(np.mean(cv, 0))  # Cost, 1st deriv
        self.cur_trajinfo_Cm(np.mean(Cm, 0))  # Cost, 2nd deriv
        self.cur_cs = cs
        # TODO: Implement policy sample costs

    def advance_timestep(self):
        """
        Move all 'cur' variables to 'prev'.
        Advance iteration
        """
        self.iteration += 1
        # TODO: Remove. This is very hacky
        for varname in self.iteration_vars:
            setattr(self, 'prev_' + varname, getattr(self, 'cur_' + varname))
            setattr(self, 'cur_' + varname, None)
