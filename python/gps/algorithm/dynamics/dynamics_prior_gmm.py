import logging
import numpy as np

from gps.utility.gmm import GMM

LOGGER = logging.getLogger(__name__)


class DynamicsPriorGMM(object):
    """
    A dynamics prior encoded as a GMM over [x_t, u_t, x_t+1] points

    See:
    S. Levine, P. Abbeel, "Learning Neural Network Policies with Guided Policy Search 
    under Unknown Dynamics", NIPS 2014
    """
    def __init__(self, min_samples_per_cluster=40, max_clusters=40, max_samples=20, strength=1.0):
        """
        Args:
            min_samples_per_cluster: Default 40
            max_clusters: Maximum number of clusters to fit. Default 40
            max_samples: Maximum number of trajectories to use for fitting the GMM at 
                any given time. Default 20
            strength: Adjust strength of prior. Default 1.0
        """
        self.X = None
        self.U = None
        self.gmm = GMM()
        self.min_samples_per_cluster = min_samples_per_cluster
        self.max_samples = max_clusters
        self.max_clusters = max_samples
        self.strength = strength

    def initial_state(self):
        """
        Return dynamics prior for initial time step
        """
        # Compute mean and covariance.
        mu0 = np.mean(self.X[:,0,:],axis=0)
        Phi = np.diag(np.var(self.X[:,0,:],axis=0))

        # Factor in multiplier.
        n0 = self.X.shape[2]*self.strength;
        m = self.X.shape[2]*self.strength;

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi*m;
        return mu0, Phi, m, n0

    def update(self,X,U):
        """
        Update prior with additional data
        Args:
            X: A N x T x dX matrix of sequential state data
            U: A N x T x dU matrix of sequential control data

        Returns: None
        """
        # Retrain GMM using additional data.
        # Constants.
        T = X.shape[1]-1; # Note that we subtract 1, since last step doesn't have dynamics.

        # Append data to dataset.
        if self.X is None:
            self.X = X
        else:
            self.X = np.concatenate([self.X, X], axis=0)

        if self.U is None:
            self.U = U
        else:
            self.U = np.concatenate([self.U, U], axis=0)

        # Remove excess samples from dataset.
        strt = max(0,self.X.shape[0]-self.max_samples+1)
        self.X = self.X[strt:,:]
        self.U = self.U[strt:,:]

        # Compute cluster dimensionality.
        Do = X.shape[2]+U.shape[2]+X.shape[2];  # TODO: Use Xtgt

        # Create dataset.
        N = self.X.shape[0]
        xux = np.reshape(np.c_[self.X[:,:T,:],self.U[:,:T,:],self.X[:,1:(T+1),:]], [T*N, Do])

        # Choose number of clusters.
        K = int(max(2,min(self.max_clusters, np.floor(float(N*T)/self.min_samples_per_cluster))))
        LOGGER.debug('Generating %d clusters for dynamics GMM.', K)

        # Update GMM.
        self.gmm.update(xux,K)

    def eval(self, Dx, Du, pts):
        """
        Evaluate prior
        Args:
            pts: A N x Dx+Du+Dx matrix
        """
        # Evaluate the dynamics prior.

        # Construct query data point by rearranging entries and adding in
        # reference.
        assert pts.shape[1] == Dx+Du+Dx

        # Perform query and fix mean.
        mu0,Phi,m,n0 = self.gmm.inference(pts)

        # Factor in multiplier.
        n0 = n0*self.strength
        m = m*self.strength

        # Multiply Phi by m (since it was normalized before).
        Phi = Phi*m
        return mu0, Phi, m, n0
