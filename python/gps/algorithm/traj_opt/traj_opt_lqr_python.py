from numpy.linalg import LinAlgError
from traj_opt import TrajOpt
import numpy as np
import scipy as sp
import logging

LOGGER = logging.getLogger(__name__)

class TrajOptLQRPython(TrajOpt):
    """LQR trajectory optimization, python implementation
    """
    def __init__(self, hyperparams):
        TrajOpt.__init__(self, hyperparams)

    def update(self):
        pass

    def estimate_cost(self, traj_distr, trajinfo):
        """
        Compute Laplace approximation to expected cost.

        Args:
            traj_distr: Linear gaussian policy object
            trajinfo:
        """

        # Constants.
        # Dx = traj_distr.K.shape[2]
        # Du = traj_distr.K.shape[1]
        T = traj_distr.K.shape[0]

        # IMPORTANT: trajinfo and traj must be constructed around the same
        # reference trajectory. This is quite important!

        # Perform forward pass (note that we repeat this here, because trajinfo may
        # have different dynamics from the ones that were used to compute the
        # distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, trajinfo)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = trajinfo.cc[t] + 0.5*np.sum(np.sum(sigma[t, :, :]*trajinfo.Cm[t, :, :])) + \
                0.5*mu[t, :].T.dot(trajinfo.Cm[t, :, :]).dot(mu[t, :]) + mu[t, :].T.dot(trajinfo.cv[t, :])
        return predicted_cost

    #TODO: Update code so that it runs. Clean args and names of variables
    def forward(self, traj_distr, trajinfo):
        """
        Forward pass - computes trajectory means and variance.
        Does NOT run yet. Also need to flip axes.

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
        sigma = np.zeros((T, dX+dU, dX+dU))
        mu = np.zeros((T, dX+dU))

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = trajinfo.x0sigma
        mu[0, idx_x] = trajinfo.x0mu

        # Perform forward pass.
        for t in range(T):
            sigma[t, :, :] = np.vstack([
                                np.hstack([sigma[t, idx_x, idx_x], sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
                                np.hstack([traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]), traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T) + traj_distr.pol_covar[t, :, :]])]
                            )
            mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])
            if t < T-1:
                sigma[t+1, idx_x, idx_x] = trajinfo.dynamics.Fd[t, :, :].dot(sigma[t, :, :]).dot(trajinfo.dynamics.Fd[t, :, :].T) + trajinfo.dynamics.dynsig[t, :, :]
                mu[t+1, idx_x] = trajinfo.dynamics.Fd[t, :, :].dot(mu[t, :]) + trajinfo.dynamics.fc[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr, trajinfo, eta):
        """
        Perform LQR backward pass. This computes a new LinearGaussianPolicy object.

        Args:
            prev_traj_distr: A Linear gaussian policy object from previous iteration
            trajinfo: A trajectory info object (need dynamics and cost matrices)
            eta: Dual variable
        Returns:
            traj_distr: New linear gaussian policy
            new_eta: Update dual variable. Updates happen if Q-function is not positive definite.
        """
        # Without GPS, simple LQR pass always converges in one iteration.

        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        traj_distr = prev_traj_distr.zeros_like()

        idx_x = slice(dX)
        idx_u = slice(dX, dX+dU)

        # Pull out cost and dynamics.
        Fd = trajinfo.dynamics.Fd
        fc = trajinfo.dynamics.fc

        # Non-SPD correction terms.
        del_ = 1e-4
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-SPD

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))

            for t in range(T-1, -1, -1):
                # Compute state-action-state function at this step.
                # Add in the cost.
                Qtt = trajinfo.Cm[t, :, :]/eta  # (X+U) x (X+U)
                Qt = trajinfo.cv[t, :]/eta  # (X+U) x 1

                # import pdb; pdb.set_trace()
                # Add in the trajectory divergence term.
                Qtt = Qtt + np.vstack([
                             np.hstack([prev_traj_distr.K[t, :, :].T.dot(prev_traj_distr.inv_pol_covar[t, :, :]).dot(prev_traj_distr.K[t, :, :]), -prev_traj_distr.K[t, :, :].T.dot(prev_traj_distr.inv_pol_covar[t, :, :])]),  # X x (X+U)
                             np.hstack([-prev_traj_distr.inv_pol_covar[t, :, :].dot(prev_traj_distr.K[t, :, :]), prev_traj_distr.inv_pol_covar[t, :, :]])  # U x (X+U)
                             ])
                Qt = Qt + np.hstack([prev_traj_distr.K[t, :, :].T.dot(prev_traj_distr.inv_pol_covar[t, :, :]).dot(prev_traj_distr.k[t, :]), -prev_traj_distr.inv_pol_covar[t, :, :].dot(prev_traj_distr.k[t, :])])

                # Add in the value function from the next time step.
                if t < T-1:
                    Qtt = Qtt + Fd[t, :, :].T.dot(Vxx[t+1, :, :]).dot(Fd[t, :, :]) #fd(:,:,t)'*Vxx(:,:,t+1)*fd(:,:,t);
                    Qt = Qt + Fd[t, :, :].T.dot(Vx[t+1, :] + Vxx[t+1, :, :].dot(fc[t, :])) #fd(:,:,t).T*(Vx(:,t+1) + Vxx(:,:,t+1)*fc(:,t));

                # Symmetrize quadratic component.
                Qtt = 0.5*(Qtt+Qtt.T)

                # Compute Cholesky decomposition of Q function action component.
                try:
                    U = sp.linalg.cholesky(Qtt[idx_u, idx_u])
                except LinAlgError as e:
                    LOGGER.info(e)
                    fail = True
                    break

                Uinv = np.linalg.inv(U)
                UTinv = np.linalg.inv(U.T)

                # Store conditional covariance and its inverse.
                traj_distr.inv_pol_covar[t, :, :] = Qtt[idx_u, idx_u]
                traj_distr.pol_covar[t, :, :] = Uinv.dot(UTinv.dot(np.eye(dU)))

                # Compute mean terms.
                traj_distr.k[t, :] = -Uinv.dot(UTinv.dot(Qt[idx_u]))
                traj_distr.K[t, :, :] = -Uinv.dot(UTinv.dot(Qtt[idx_u, idx_x]))

                # Compute value function.
                Vxx[t, :, :] = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vx[t, :] = Qt[idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            # Increment eta on non-PD Q-function
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.info('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure
                if eta >= 1e16:
                    if np.any(np.any(np.any(np.isnan(Fd)))) or np.any(np.any(np.isnan(fc))):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise Exception('Failed to find positive definite LQR solution even for very large eta (check that dynamics and cost are reasonably well conditioned)!')
        return traj_distr, eta