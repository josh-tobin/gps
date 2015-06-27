import numpy as np

from traj_opt import TrajOpt
from traj_opt_util import bracketing_line_search, traj_distr_kl


# Constants - TODO: put in a different file?
DGD_MAX_ITER = 50
THRESHA = 1e-4
THRESHB = 1e-3

class TrajOptLQRPython(TrajOpt):
    """LQR trajectory optimization, python implementation
    """
    def __init__(self, hyperparams):
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
        kl_step = self._hyperparams['kl_step']*traj_distr.step_mult

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
            if (abs(kl_div-kl_step*T) < 0.1*kl_step*T or
                    (itr >= 20 and kl_div < kl_step*T)):
                # TODO - Log/print debug info here
                break

            # Adjust eta using bracketing line search
            line_search_data, eta = bracketing_line_search(line_search_data,
                                                           kl_div-kl_step*T,
                                                           new_eta,
                                                           min_eta)

            # Convergence check - dual variable change when min_eta hit
            if (abs(prev_eta-eta) < THRESHA and
                    eta == max(min_eta, self._hyperparams['min_eta'])):
                # TODO - Log/print debug info here
                break

            # Convergence check - constraint satisfaction, kl not changing much
            if (itr > 2 and abs(kl_div-prev_kl_div) < THRESHB and
                    kl_div < kl_step*T):
                # TODO - Log/print debug info here
                break

            prev_kl_div = kl_div
            prev_eta = eta
            # TODO - Log/print progress/debug info

        if kl_div > kl_step*T and abs(kl_div-kl_step*T) > 0.1*kl_step*T:
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
        # dX = traj_distr.K.shape[2]
        # dU = traj_distr.K.shape[1]
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
                sigma[t+1, idx_x, idx_x] = trajinfo.dynamics.Fm[t, :, :].dot(sigma[t, :, :]).dot(trajinfo.dynamics.Fm[t, :, :].T) + trajinfo.dynamics.dynsig[t, :, :]
                mu[t+1, idx_x] = trajinfo.dynamics.Fm[t, :, :].dot(mu[t, :]) + trajinfo.dynamics.fv[t, :]
        return mu, sigma

    #TODO: Update code so that it runs. Clean args and names of variables
    def backward(self, traj_distr, prevtraj_distr, trajinfo, eta):
        """
        Args:
            TODO
        Returns:
            traj_distr: Updated K, k, PSig, etc.
            new_eta: New dual variable (eta)
        """
        # Perform LQR backward pass to compute new policy.

        # Without GPS, simple LQR pass always converges in one iteration.

        # Constants.
        dX = self.dX
        dU = self.dU
        T = self.T
        idx_x = slice(self.dX)
        idx_u = slice(self.dX, self.dX+self.dU)

        # Pull out cost and dynamics.
        Fm = trajinfo.dynamics.Fm
        fv = trajinfo.dynamics.fv

        # Non-SPD correction terms.
        eta0 = eta
        del0 = 1e-4
        del_ = 0

        # Run dynamic programming.
        fail = True
        while fail:
            # Allocate.
            Vxx = np.zeros((dX,dX,T))
            Vx = np.zeros((dX,T))

            # Don't fail.
            fail = 0
            for t in range(T-1, -1, -1):
                # Compute state-action-state function at this step.
                # Add in the cost.
                Qtt = trajinfo.Cm[:,:,t]/eta
                Qt = trajinfo.cv[:,t]/eta

                # Add in the trajectory divergence term.
                Qtt = Qtt + np.vstack([
                             np.hstack([prevtraj_distr.K[:,:,t].T.dot(prevtraj_distr.invPSig[:,:,t]).dot(prevtraj_distr.K[:,:,t]), -prevtraj_distr.K[:,:,t].T.dot(prevtraj_distr.invPSig[:,:,t])]),
                             np.hstack([-prevtraj_distr.invPSig[:,:,t].dot(prevtraj_distr.K[:,:,t]), prevtraj_distr.invPSig[:,:,t]])
                             ])
                Qt = Qt + np.hstack([prevtraj_distr.K[:,:,t].T.dot(prevtraj_distr.invPSig[:,:,t]).dot(prevtraj_distr.k[:,t]), -prevtraj_distr.invPSig[:,:,t].dot(prevtraj_distr.k[:,t])])

                # Add in the value function from the next time step.
                if t < T-1:
                    Qtt = Qtt + Fm[:,:,t].T.dot(Vxx[:,:,t+1]).dot(Fm[:,:,t]) #Fm(:,:,t)'*Vxx(:,:,t+1)*Fm(:,:,t);
                    Qt = Qt + Fm[:,:,t].T.dot(Vx[:,t+1] + Vxx[:,:,t+1].dot(fv[:,t])) #Fm(:,:,t).T*(Vx(:,t+1) + Vxx(:,:,t+1)*fv(:,t));

                # Symmetrize quadratic component.
                Qtt = 0.5*(Qtt+Qtt.T)

                # Compute Cholesky decomposition of Q function action component.
                L = np.linalg.cholesky(Qtt[idx_u, idx_u])

                #TODO: Is matlab backslash (\) equivalent to inverse?
                Linv = np.linalg.inv(L)
                LTinv = np.linalg.inv(L.T)

                # Store conditional covariance and its inverse.
                traj_distr.invPSig[:, :, t] = Qtt[idx_u, idx_u]
                traj_distr.PSig[:, :, t] = Linv.dot(LTinv.dot(np.eye(dU)))

                # Compute mean terms.
                traj_distr.k[:, t] = -Linv.dot(LTinv.dot(Qt[idx_u, 1]))
                traj_distr.K[:, :, t] = -Linv.dot(LTinv.dot(Qtt[idx_u, idx_x]))

                # Compute value function.
                Vxx[:,:,t] = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.K[:, :, t])
                Vx[:,t] = Qt[idx_x, 1] + Qtt[idx_x, idx_u].dot(traj_distr.k[:, t])
                Vxx[:,:,t] = 0.5 * (Vxx[:,:,t] + Vxx[:,:,t].T)
            # Increment eta if failed.
            # TODO: Eta update on non-SPD
            """
            if fail,
                del_ = max(del0,del*2);
                eta = eta0 + del;
                if #~isnan(algorithm.params.fid_debug),fprintf(algorithm.params.fid_debug,'Increasing eta: %f\n',eta);end;
                if eta >= 1e16,
                    if any(any(any(isnan(Fm)))) || any(any(isnan(fv))),
                        error('NaNs encountered in dynamics!');
                    end;
                    error('Failed to find positive definite LQR solution even for very large eta (check that dynamics and cost are reasonably well conditioned)!');
                end;
            end;
            """
