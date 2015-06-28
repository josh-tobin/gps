from numpy.linalg import LinAlgError
import numpy as np
import scipy as sp
import logging
import copy

from config import traj_opt_lqr
from traj_opt import TrajOpt
from traj_opt_util import bracketing_line_search, traj_distr_kl


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
        TrajOpt.__init__(self, hyperparams)

    # TODO - traj_distr and prevtraj_distr shouldn't be arguments - should exist in self.
    def update(self, traj_distr, prevtraj_distr):
        """Run dual gradient decent to optimize trajectories."""

        # Constants
        dX = self.dX
        dU = self.dU
        T = self.T

        # Set KL-divergence step size (epsilon)
        # TODO - traj_distr.step_mult needs to exist somewhere
        kl_step = self._hyperparams['kl_step'] * traj_distr.step_mult

        line_search_data = {}
        eta = traj_distr.eta
        prev_eta = eta
        min_eta = -np.Inf

        for itr in range(DGD_MAX_ITER):
            new_traj_distr, new_eta = self.backward(traj_distr,
                                                    prevtraj_distr,
                                                    trajinfo,
                                                    eta)
            new_mu, new_sigma = self.forward(new_traj_distr, trajinfo)

            # Update min eta if we had a correction after running backward
            if new_eta > eta:
                min_eta = new_eta

            # Compute KL divergence between previous and new distribuition
            kl_div = traj_distr_kl(new_mu, new_sigma, new_traj_distr,
                                   prev_traj_distr)

            # TODO - Previously have stored lastklstep here, but that is only
            # used in TrajOptBADMM, TrajOptADMM, and TrajOptCGPS

            # Main convergence check - constraint satisfaction
            if (abs(kl_div - kl_step * T) < 0.1 * kl_step * T or
                    (itr >= 20 and kl_div < kl_step * T)):
                # TODO - Log/print debug info here
                break

            # Adjust eta using bracketing line search
            line_search_data, eta = bracketing_line_search(line_search_data,
                                                           kl_div - kl_step * T,
                                                           new_eta,
                                                           min_eta)

            # Convergence check - dual variable change when min_eta hit
            if (abs(prev_eta - eta) < THRESHA and
                        eta == max(min_eta, self._hyperparams['min_eta'])):
                # TODO - Log/print debug info here
                break

            # Convergence check - constraint satisfaction, kl not changing much
            if (itr > 2 and abs(kl_div - prev_kl_div) < THRESHB and
                        kl_div < kl_step * T):
                # TODO - Log/print debug info here
                break

            prev_kl_div = kl_div
            prev_eta = eta
            # TODO - Log/print progress/debug info

        if kl_div > kl_step * T and abs(kl_div - kl_step * T) > 0.1 * kl_step * T:
            fprintf()
            # TODO Log warning - "Final KL divergence after DGD convergence is too high"

            # TODO - store new_traj_distr somewhere (in self?)


    def estimate_cost(self, traj_distr, trajinfo):
        """
        Compute Laplace approximation to expected cost.

        Args:
            traj_distr: Linear gaussian policy object
            trajinfo:
        """

        # Constants.
        T = traj_distr.T

        # Perform forward pass (note that we repeat this here, because trajinfo may
        # have different dynamics from the ones that were used to compute the
        # distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, trajinfo)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = trajinfo.cc[t] + 0.5 * np.sum(np.sum(sigma[t, :, :] * trajinfo.Cm[t, :, :])) + \
                                0.5 * mu[t, :].T.dot(trajinfo.Cm[t, :, :]).dot(mu[t, :]) + mu[t, :].T.dot(
                trajinfo.cv[t, :])
        return predicted_cost

    def forward(self, traj_distr, trajinfo):
        """
        Perform LQR forward pass.
        Computes state-action marginals from dynamics and policy.

        Args:
            traj_distr: A linear gaussian policy object
            trajinfo: A traj info object
        Returns:
            mu: T x dX
            sigma: T x dX x dX
        """
        # Compute state-action marginals from specified conditional parameters and
        # current trajinfo.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX + dU, dX + dU))
        mu = np.zeros((T, dX + dU))

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = trajinfo.x0sigma
        mu[0, idx_x] = trajinfo.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([sigma[t, idx_x, idx_x], sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
                np.hstack([traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                           traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                               traj_distr.K[t, :, :].T) + traj_distr.pol_covar[t, :, :]])]
            )
            mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])
            if t < T - 1:
                sigma[t + 1, idx_x, idx_x] = trajinfo.dynamics.Fm[t, :, :].dot(sigma[t, :, :]).dot(
                    trajinfo.dynamics.Fm[t, :, :].T) + trajinfo.dynamics.dynsig[t, :, :]
                mu[t + 1, idx_x] = trajinfo.dynamics.Fm[t, :, :].dot(mu[t, :]) + trajinfo.dynamics.fv[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr, trajinfo, eta):
        """
        Perform LQR backward pass.
        This computes a new LinearGaussianPolicy object.

        Args:
            prev_traj_distr: A Linear gaussian policy object from previous iteration
            trajinfo: A trajectory info object (need dynamics and cost matrices)
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
        Fm = trajinfo.dynamics.Fm
        fv = trajinfo.dynamics.fv

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
                Qtt = trajinfo.Cm[t, :, :] / eta  # (X+U) x (X+U)
                Qt = trajinfo.cv[t, :] / eta  # (X+U) x 1

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
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not symmetric positive definite.
                    LOGGER.debug(e)
                    fail = True
                    break

                Uinv = np.linalg.inv(U)
                UTinv = np.linalg.inv(U.T)

                # Store conditional covariance, its inverse, and cholesky
                traj_distr.inv_pol_covar[t, :, :] = Qtt[idx_u, idx_u]
                traj_distr.pol_covar[t, :, :] = Uinv.dot(UTinv.dot(np.eye(dU)))
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(traj_distr.pol_covar[t, :, :])

                # Compute mean terms.
                traj_distr.k[t, :] = -Uinv.dot(UTinv.dot(Qt[idx_u]))
                traj_distr.K[t, :, :] = -Uinv.dot(UTinv.dot(Qtt[idx_u, idx_x]))

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
