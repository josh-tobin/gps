from numpy.linalg import LinAlgError
import numpy as np
import scipy as sp
import logging
import copy

from config import traj_opt_lqr
from traj_opt import TrajOpt
from traj_opt_util import LineSearch, traj_distr_kl


# Constants - TODO: put in a different file?
DGD_MAX_ITER = 50
THRESHA = 1e-4
THRESHB = 1e-3
LOGGER = logging.getLogger(__name__)


class TrajOptLQRPython(TrajOpt):
    """LQR trajectory optimization, python implementation
    """

    def __init__(self, hyperparams):
        config = copy.deepcopy(traj_opt_lqr)
        config.update(hyperparams)
        TrajOpt.__init__(self, config)

    # TODO - traj_distr and prevtraj_distr shouldn't be arguments - should exist in self.
    def update(self, T, step_mult, eta, traj_info, prev_traj_distr):
        """Run dual gradient decent to optimize trajectories."""

        # Constants

        # Set KL-divergence step size (epsilon)
        # TODO - traj_distr.step_mult needs to exist somewhere
        kl_step = self._hyperparams['kl_step'] * step_mult

        line_search = LineSearch(self._hyperparams['min_eta'])
        prev_eta = eta
        min_eta = -np.Inf

        for itr in range(DGD_MAX_ITER):
            new_traj_distr, new_eta = self.backward(prev_traj_distr,
                                                    traj_info,
                                                    eta)
            new_mu, new_sigma = self.forward(new_traj_distr, traj_info)

            # Update min eta if we had a correction after running backward
            if new_eta > eta:
                min_eta = new_eta

            # Compute KL divergence between previous and new distribuition
            kl_div = traj_distr_kl(new_mu, new_sigma, new_traj_distr,
                                   prev_traj_distr)

            # TODO - Previously have stored lastklstep here, but that is only
            # used in TrajOptBADMM, TrajOptADMM, and TrajOptCGPS

            # Main convergence check - constraint satisfaction
            if (abs(kl_div - kl_step*T) < 0.1*kl_step*T or
                    (itr >= 20 and kl_div < kl_step*T)):
                LOGGER.debug("Iteration %i, KL: %f / %f converged",
                             itr, kl_div, kl_step*T)
                break

            # Adjust eta using bracketing line search
            eta = line_search.bracketing_line_search(kl_div - kl_step*T,
                                                     new_eta,
                                                     min_eta)

            # Convergence check - dual variable change when min_eta hit
            if (abs(prev_eta - eta) < THRESHA and
                        eta == max(min_eta, self._hyperparams['min_eta'])):
                LOGGER.debug("Iteration %i, KL: %f / %f converged (eta limit)",
                             itr, kl_div, kl_step*T)
                break

            # Convergence check - constraint satisfaction, kl not changing much
            if (itr > 2 and abs(kl_div - prev_kl_div) < THRESHB and
                        kl_div < kl_step*T):
                LOGGER.debug("Iteration %i, KL: %f / %f converged (no change)",
                             itr, kl_div, kl_step*T)
                break

            prev_kl_div = kl_div
            LOGGER.debug('Iteration %i, KL: %f / %f eta: %f -> %f',
                         itr, kl_div, kl_step*T, prev_eta, eta)
            prev_eta = eta

        if kl_div > kl_step*T and abs(kl_div - kl_step*T) > 0.1*kl_step*T:
            LOGGER.warning("Final KL divergence after DGD convergence is too high")

            # TODO - store new_traj_distr somewhere (in self?)
        return new_traj_distr, eta

    def estimate_cost(self, traj_distr, traj_info):
        """
        Compute Laplace approximation to expected cost.

        Args:
            traj_distr: Linear gaussian policy object
            traj_info:
        """

        # Constants.
        T = traj_distr.T

        # Perform forward pass (note that we repeat this here, because traj_info may
        # have different dynamics from the ones that were used to compute the
        # distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + 0.5 * np.sum(np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :])) + \
                                0.5 * mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + mu[t, :].T.dot(
                traj_info.cv[t, :])
        return predicted_cost

    def forward(self, traj_distr, traj_info):
        """
        Perform LQR forward pass.
        Computes state-action marginals from dynamics and policy.

        Args:
            traj_distr: A linear gaussian policy object
            traj_info: A traj info object
        Returns:
            mu: T x dX
            sigma: T x dX x dX
        """
        # Compute state-action marginals from specified conditional parameters and
        # current traj_info.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX + dU, dX + dU))
        mu = np.zeros((T, dX + dU))

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([sigma[t, idx_x, idx_x], sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
                np.hstack([traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                           traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                               traj_distr.K[t, :, :].T) + traj_distr.pol_covar[t, :, :]])]
            )
            mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])
            if t < T - 1:
                sigma[t + 1, idx_x, idx_x] = traj_info.dynamics.Fm[t, :, :].dot(sigma[t, :, :]).dot(
                    traj_info.dynamics.Fm[t, :, :].T) + traj_info.dynamics.dyn_covar[t, :, :]
                mu[t + 1, idx_x] = traj_info.dynamics.Fm[t, :, :].dot(mu[t, :]) + traj_info.dynamics.fv[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta):
        """
        Perform LQR backward pass.
        This computes a new LinearGaussianPolicy object.

        Args:
            prev_traj_distr: A Linear gaussian policy object from previous iteration
            traj_info: A trajectory info object (need dynamics and cost matrices)
            eta: Dual variable
        Returns:
            traj_distr: New linear gaussian policy
            new_eta: Updated dual variable. Updates happen if Q-function is not symmetric positive definite.
        """
        # Without GPS, simple LQR pass always converges in one iteration.

        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        traj_distr = prev_traj_distr.nans_like()

        idx_x = slice(dX)
        idx_u = slice(dX, dX + dU)

        # Pull out cost and dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self._hyperparams['del0']
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric positive definite

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))

            for t in range(T - 1, -1, -1):
                # Compute state-action-state function at this step.
                # Add in the cost.
                Qtt = traj_info.Cm[t, :, :] / eta  # (X+U) x (X+U)
                Qt = traj_info.cv[t, :] / eta  # (X+U) x 1

                # Add in the trajectory divergence term.
                Qtt = Qtt + np.vstack([
                    np.hstack([prev_traj_distr.K[t, :, :].T.dot(prev_traj_distr.inv_pol_covar[t, :, :]).dot(
                        prev_traj_distr.K[t, :, :]),
                               -prev_traj_distr.K[t, :, :].T.dot(prev_traj_distr.inv_pol_covar[t, :, :])]),  # X x (X+U)
                    np.hstack([-prev_traj_distr.inv_pol_covar[t, :, :].dot(prev_traj_distr.K[t, :, :]),
                               prev_traj_distr.inv_pol_covar[t, :, :]])  # U x (X+U)
                ])
                Qt = Qt + np.hstack([prev_traj_distr.K[t, :, :].T.dot(prev_traj_distr.inv_pol_covar[t, :, :]).dot(
                    prev_traj_distr.k[t, :]), -prev_traj_distr.inv_pol_covar[t, :, :].dot(prev_traj_distr.k[t, :])])

                # Add in the value function from the next time step.
                if t < T - 1:
                    Qtt = Qtt + Fm[t, :, :].T.dot(Vxx[t + 1, :, :]).dot(Fm[t, :, :])
                    Qt = Qt + Fm[t, :, :].T.dot(Vx[t + 1, :] + Vxx[t + 1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt = 0.5 * (Qtt + Qtt.T)

                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(Qtt[idx_u, idx_u])
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not symmetric positive definite.
                    LOGGER.debug('LinAlgError:%s' % e)
                    fail = True
                    break


                # Store conditional covariance, its inverse, and cholesky
                traj_distr.inv_pol_covar[t, :, :] = Qtt[idx_u, idx_u]
                traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True))
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(traj_distr.pol_covar[t, :, :])

                # Compute mean terms.
                traj_distr.k[t, :] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, Qt[idx_u], lower=True))
                traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(L, Qtt[idx_u, idx_x], lower=True))

                # Compute value function.
                Vxx[t, :, :] = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vx[t, :] = Qt[idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            # Increment eta on non-SPD Q-function
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure
                if eta >= 1e16:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError(
                        'Failed to find positive definite LQR solution even for very large eta ' +
                        '(check that dynamics and cost are reasonably well conditioned)!')
        return traj_distr, eta
