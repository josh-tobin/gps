from traj_opt import TrajOpt
import numpy as np


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
        Dx = traj_distr.K.shape[2]
        Du = traj_distr.K.shape[1]
        T = traj_distr.K.shape[0]

        # IMPORTANT: trajinfo and traj must be constructed around the same
        # reference trajectory. This is quite important!

        # Perform forward pass (note that we repeat this here, because trajinfo may
        # have different dynamics from the ones that were used to compute the
        # distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, trajinfo.dynamics, trajinfo.x0mu, trajinfo.x0sigma)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = trajinfo.cc[t] + 0.5*np.sum(np.sum(sigma[t, :, :]*trajinfo.Cm[t, :, :])) + \
                0.5*mu[t, :].T.dot(trajinfo.Cm[t, :, :]).dot(mu[t, :]) + mu[t, :].T.dot(trajinfo.cv[t, :])
        return predicted_cost

    #TODO: Update code so that it runs. Clean args and names of variables
    def forward(self, traj_distr, dynamics, x0mu, x0sigma):
        """
        Forward pass - computes trajectory means and variance.
        Does NOT run yet. Also need to flip axes.

        Args:
            traj_distr:
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
        sigma[0, idx_x, idx_x] = x0sigma
        mu[0, idx_x] = x0mu

        # Perform forward pass.
        for t in range(T):
            sigma[t, :, :] = np.vstack([
                                np.hstack([sigma[t, idx_x, idx_x], sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)]),
                                np.hstack([traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]), traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(traj_distr.K[t, :, :].T) + traj_distr.pol_covar[t, :, :]])]
                            )
            mu[t, :] = np.hstack([mu[t, idx_x], traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]])
            if t < T-1:
                sigma[t+1, idx_x, idx_x] = dynamics.fd[t, :, :].dot(sigma[t, :, :]).dot(dynamics.fd[t, :, :].T) + dynamics.dynsig[t, :, :]
                mu[t+1, idx_x] = dynamics.fd[t, :, :].dot(mu[t, :]) + dynamics.fc[t, :]
        return mu, sigma

    #TODO: Update code so that it runs. Clean args and names of variables
    def backward(self, mu, sigma, traj_distr, prevtraj, dynamics, cv, Cm, eta):
        # Perform LQR backward pass to compute new policy.

        # Without GPS, simple LQR pass always converges in one iteration.

        # Constants.
        Dx = self.Dx
        Du = self.Du
        T = self.T
        idx_x = slice(self.Dx)
        idx_u = slice(self.Dx, self.Dx+self.Du)

        # Pull out cost and dynamics.
        fd = dynamics.fd
        fc = dynamics.fc

        # Non-SPD correction terms.
        eta0 = eta
        del0 = 1e-4
        del_ = 0

        # Run dynamic programming.
        fail = True
        while fail:
            # Allocate.
            Vxx = np.zeros((Dx,Dx,T))
            Vx = np.zeros((Dx,T))

            # Don't fail.
            fail = 0
            for t in range(T-1, -1, -1):
                # Compute state-action-state function at this step.
                # Add in the cost.
                Qtt = Cm[:,:,t]/eta
                Qt = cv[:,t]/eta

                # Add in the trajectory divergence term.
                Qtt = Qtt + np.vstack([
                             np.hstack([prevtraj.K[:,:,t].T.dot(prevtraj.invPSig[:,:,t]).dot(prevtraj.K[:,:,t]), -prevtraj.K[:,:,t].T.dot(prevtraj.invPSig[:,:,t])]),
                             np.hstack([-prevtraj.invPSig[:,:,t].dot(prevtraj.K[:,:,t]), prevtraj.invPSig[:,:,t]])
                             ])
                Qt = Qt + np.hstack([prevtraj.K[:,:,t].T.dot(prevtraj.invPSig[:,:,t]).dot(prevtraj.k[:,t]), -prevtraj.invPSig[:,:,t].dot(prevtraj.k[:,t])])

                # Add in the value function from the next time step.
                if t < T-1:
                    Qtt = Qtt + fd[:,:,t].T.dot(Vxx[:,:,t+1]).dot(fd[:,:,t]) #fd(:,:,t)'*Vxx(:,:,t+1)*fd(:,:,t);
                    Qt = Qt + fd[:,:,t].T.dot(Vx[:,t+1] + Vxx[:,:,t+1].dot(fc[:,t])) #fd(:,:,t).T*(Vx(:,t+1) + Vxx(:,:,t+1)*fc(:,t));

                # Symmetrize quadratic component.
                Qtt = 0.5*(Qtt+Qtt.T)

                # Compute Cholesky decomposition of Q function action component.
                L = np.linalg.cholesky(Qtt[idx_u, idx_u])

                #TODO: Is matlab backslash (\) equivalent to inverse?
                Linv = np.linalg.inv(L)
                LTinv = np.linalg.inv(L.T)

                # Store conditional covariance and its inverse.
                traj_distr.invPSig[:, :, t] = Qtt[idx_u, idx_u]
                traj_distr.PSig[:, :, t] = Linv.dot(LTinv.dot(np.eye(Du)))

                # Compute mean terms.
                traj_distr.k[:, t] = -Linv.dot(LTinv.dot(Qt[idx_u, 1]))
                traj_distr.K[:, :, t] = -Linv.dot(LTinv.dot(Qtt[idx_u, idx_x]))

                # Compute value function.
                Vxx[:,:,t] = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(traj_distr.K[:, :, t])
                Vx[:,t] = Qt[idx_x, 1] + Qtt[idx_x, idx_u].dot(traj_distr.k[:, t])
                Vxx[:,:,t] = 0.5 * (Vxx[:,:,t] + Vxx[:,:,t].T)
            # Increment eta if failed.
            """
            if fail,
                del_ = max(del0,del*2);
                eta = eta0 + del;
                if #~isnan(algorithm.params.fid_debug),fprintf(algorithm.params.fid_debug,'Increasing eta: %f\n',eta);end;
                if eta >= 1e16,
                    if any(any(any(isnan(fd)))) || any(any(isnan(fc))),
                        error('NaNs encountered in dynamics!');
                    end;
                    error('Failed to find positive definite LQR solution even for very large eta (check that dynamics and cost are reasonably well conditioned)!');
                end;
            end;
            """