from traj_opt import TrajOpt
import numpy as np


class TrajOptLQRPython(TrajOpt):
    """LQR trajectory optimization, python implementation
    """
    def __init__(self, hyperparams, dynamics):
        TrajOpt.__init__(self, hyperparams, dynamics)
        #TODO: Fill in
        self.Dx = 0
        self.Du = 0
        self.T = 0

    def update(self):
        pass

    #TODO: Update code so that it runs. Clean args and names of variables
    def estimate_cost(self, traj_distr, dynamics, initial_traj, cc, cv, Cm):
        """
        Compute Laplace approximation to expected cost.

        Args:
            traj_distr: Linear gaussian
            dynamics: Linear dynamics
            initial_traj: Initial mu, sigma
            cc: Cost
            cv: Cost 1st derivative
            Cm: Cost 2nd derivative
        """

        # Constants.
        Dx = traj_distr.K.shape(2)
        Du = traj_distr.K.shape(1)
        T = traj_distr.K.shape(0)

        # IMPORTANT: trajinfo and traj must be constructed around the same
        # reference trajectory. This is quite important!

        # Perform forward pass (note that we repeat this here, because trajinfo may
        # have different dynamics from the ones that were used to compute the
        # distribution already saved in traj).
        [sigma,mu] = self.forward(traj_distr, dynamics, initial_traj)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = cc(t) + 0.5*sum(sum(sigma[:,:,t]*Cm[:,:,t])) + \
                0.5*mu[:,t].T*Cm[:,:,t]*mu[:,t] + mu[:,t].T*cv[:,t]
        return predicted_cost

    #TODO: Update code so that it runs. Clean args and names of variables
    def forward(self, traj_distr, dynamics, initial_traj):
        """
        Forward pass - computes trajectory means and variance.
        Does NOT run yet. Also need to flip axes.
        """
        # Compute state-action marginals from specified conditional parameters and
        # current trajinfo.

        # Constants.
        idx_x = slice(self.Dx)

        # Allocate space.
        sigma = np.zeros((self.Dx+self.Du,self.Dx+self.Du,self.T))
        mu = np.zeros((self.Dx+self.Du,self.T))

        # Set initial covariance (initial mu is always zero).
        sigma[idx_x, idx_x, 0] = initial_traj.x0sigma
        mu[idx_x, 0] = initial_traj.x0mu

        # Perform forward pass.
        for t in range(self.T):
            sigma[:,:,t] = np.vstack([
                                np.hstack([sigma[idx_x, idx_x, t], sigma[idx_x, idx_x, t].dot(traj_distr.K[:,:,t].T)]),
                                np.hstack([traj_distr.K[:,:,t].dot(sigma[idx_x, idx_x, t]), traj_distr.K[:,:,t].dot(sigma[idx_x, idx_x, t]).dot(traj_distr.K[:,:,t].T) + traj_distr.PSig[:,:,t]])]
                            )
            mu[:,t] = np.vstack([mu[idx_x, t], traj_distr.K[:,:,t].dot(mu[idx_x,t]) + traj_distr.k[:,t]])
            if t < self.T-1:
                sigma[idx_x, idx_x, t+1] = dynamics.fd[:,:,t].dot(sigma[:,:,t]).dot(dynamics.fd[:,:,t].T) + dynamics.dynsig[:,:,t]
                mu[idx_x, t+1] = dynamics.fd[:,:,t]*mu[:,t] + dynamics.fc[:,t]
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