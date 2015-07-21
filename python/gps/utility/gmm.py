import numpy as np
import scipy as sp
import scipy.linalg
import logging

LOGGER = logging.getLogger(__name__)

def logsum(vec,axis=0, keepdims=True):
    maxv = np.max(vec,axis=axis, keepdims=keepdims)
    maxv[maxv == -float('inf')] = 0;
    return np.log(np.sum(np.exp(vec-maxv),axis=axis, keepdims=keepdims)) + maxv;

class GMM(object):
    """ Gaussian Mixture Model """
    def __init__(self, init_sequential=False, eigreg=False):
        self.init_sequential = init_sequential
        self.eigreg = eigreg

    def inference(self, pts):
        """ 
        Evaluate dynamics prior 
        Args:
            pts: A DxN array of points
        """
        # Compute posterior cluster weights.
        logwts = self.clusterwts(pts)

        # Compute posterior mean and covariance.
        mu0,Phi = self.moments(logwts)

        # Set hyperparameters.
        m = self.N;
        n0 = m - 2 - mu0.shape[0]

        # Normalize.
        m = float(m)/self.N
        n0 = float(n0)/self.N
        return mu0, Phi, m, n0

    def estep(self, data):
        """
        Compute log observation probabilities under GMM.
        Args:
            data: A DxN array of points
        Returns:
            logobs: A KxN array of log probabilities (for each point on each cluster)
        """

        # Constants.
        K = self.sigma.shape[2]
        Di = data.shape[0]
        N = data.shape[1]

        # Compute probabilities.
        mu = self.mu[0:Di, :]
        mu_expand = np.expand_dims(np.expand_dims(mu, axis=1), axis=1)
        assert mu_expand.shape == (Di,1,1,K)
        # Calculate for each point distance to each cluster
        data_expand = np.tile(data, [K,1,1,1]).transpose([2,3,1,0])
        diff = data_expand - np.tile(mu_expand, [1,N,1,1])
        assert diff.shape == (Di,N,1,K)
        Pdiff = np.zeros_like(diff)
        cconst = np.zeros((1,1,1,K))

        for i in range(K):
            U = sp.linalg.cholesky(self.sigma[:Di, :Di, i])
            Pdiff[:,:,0,i] = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, diff[:,:,0,i], lower=True))
            cconst[0,0,0,i] = -np.sum(np.log(np.diag(U))) - 0.5*Di*np.log(2*np.pi)

        logobs = -0.5*np.sum(diff*Pdiff,axis=0, keepdims=True)+cconst
        assert logobs.shape == (1, N, 1, K)
        logobs = logobs[0,:,0,:].T + self.logmass
        return logobs

    def moments(self, logwts):
        """ 
        Compute the moments of the cluster mixture with specified weights.
        Args:
            logwts: A Kx1 array of log cluster probabilities

        Returns:
            mu: A (D,) mean vector
            sigma: A DxD covariance matrix
        """

        # Exponentiate.
        wts = np.exp(logwts)

        # Compute overall mean.
        mu = np.sum(self.mu*wts.T,axis=1)

        # Compute overall covariance.
        # For some reason this version works way better than the "right" one...
        # Could we be computing xxt wrong?
        diff = self.mu-np.expand_dims(mu, axis=1)
        diff_expand = np.expand_dims(diff,axis=1)*np.expand_dims(diff,axis=0)
        wts_expand = np.expand_dims(wts.T,axis=0)
        sigma = np.sum((self.sigma + diff_expand)*wts_expand,axis=2)

        return mu, sigma

    def clusterwts(self, data):
        """
        Compute cluster weights for specified points under GMM.

        Args:
            data: A DxN array of points
        Return:
            A Kx1 array of average cluster log probabilities
        """

        # Compute probability of each point under each cluster.
        logobs = self.estep(data)

        # Renormalize to get cluster weights.
        logwts = logobs-logsum(logobs,axis=0)

        # Average the cluster probabilities.
        logwts = logsum(logwts,axis=1) - np.log(data.shape[1])
        return logwts

    def update(self, data, K, iterations=100):
        """
        Run EM to update clusters

        Args:
            data: A D x N data matrix, where N = number of data points
            K: Number of clusters to use
        """
        # Constants.
        Do = data.shape[0]
        N = data.shape[1];

        #TODO: Check if we are warmstarting. 
        if True: #~gmm.params.warmstart || isempty(gmm.sigma) || K ~= size(gmm.sigma,3),
            # Initialization.
            self.sigma = np.zeros((Do,Do,K))
            self.prec = np.zeros((Do,Do,K))
            self.xxt = np.zeros((Do,Do,K))
            self.mu = np.zeros((Do,K))
            self.logmass = np.log(1.0/K)*np.ones((K,1))
            self.mass = (1.0/K)*np.ones((K,1))
            self.logdet = np.zeros((K,1))
            self.N = data.shape[1];
            self.priorw = 1e-3;
            self.priorx = np.mean(data,axis=1);
            self.priorxxt = (1.0/N)*(data.dot(data.T)) + 1e-1*np.eye(Do);
            N = self.N;

            # Set initial cluster indices.
            if not self.init_sequential:
                cidx = np.random.randint(0,K,size=(1,N));
            else:
                raise NotImplementedError()
                """
                cidx = zeros(T,N/T);
                split = floor(linspace(1,T+1,K+1));
                for i=1:K,
                    tstrt = split(i);
                    tend = min(split(i+1)-1,T);
                    tidx = tstrt:tend;
                    cidx(tidx,:) = i;
                end;
                cidx = reshape(cidx,[1 N]);
                """

            # Initialize.
            for i in range(K):
                cluster_idx = (cidx==i)[0]
                mu = np.mean(data[:, cluster_idx],axis=1)
                diff = (data[:, cluster_idx].T-mu).T
                sigma = (1.0/K)*(diff.dot(diff.T));
                self.mu[:,i] = mu;
                self.sigma[:,:,i] = sigma + np.eye(Do)*2e-6;

        prevll = -float('inf')
        for itr in range(iterations):
            LOGGER.debug('Beginning GMM EM iteration %d/%d', itr, self.iterations)
            # E-step: compute cluster probabilities.
            logobs = self.estep(data);

            # Compute log-likelihood.
            ll = np.sum(logsum(logobs,axis=0));
            LOGGER.debug('GMM Log likelihood: %f', ll)
            if np.abs(ll-prevll) < 1e-2:
                LOGGER.debug('GMM Converged on itr=%d/%d', itr, self.iterations)
                break
            prevll = ll;

            # Renormalize to get cluster weights.
            logw = logobs-logsum(logobs,axis=0);
            assert logw.shape == (K, N)
            wc = np.exp(logw);

            # Renormalize againt to get weights for refitting clusters.
            logwn = logw-logsum(logw,axis=1);
            assert logwn.shape == (K, N)
            w = np.exp(logwn);

            # M-step: update clusters.
            # Fit cluster mass.
            self.logmass = logsum(logw,axis=1);
            self.logmass =self.logmass - logsum(self.logmass,axis=0);
            assert self.logmass.shape == (K, 1)
            self.mass = np.exp(self.logmass);
            # Reboot small clusters.
            w[(self.mass < (1.0/K)*1e-4)[:,0],:] = 1.0/N;
            # Fit cluster means.
            w_expand = np.expand_dims(w,axis=2).transpose([2,1,0])
            data_expand = np.expand_dims(data, axis=2)
            self.mu = np.sum(w_expand*data_expand,axis=1);
            # Fit covariances.
            wdata = data_expand*np.sqrt(w_expand);
            assert wdata.shape == (Do, N, K)
            for i in range(K):
                # Compute weighted outer product.
                XX = wdata[:,:,i].dot(wdata[:,:,i].T)
                self.xxt[:,:,i] = 0.5*(XX+XX.T)
                self.sigma[:,:,i] = XX - np.outer(self.mu[:,i], self.mu[:,i])

                if self.eigreg: # Use eigenvalue regularization.
                    raise NotImplementedError()
                    """
                    self.sigma[:,:,i] = 0.5*(self.sigma[:,:,i]+self.sigma[:,:,i].T)
                    [val, vec] = np.linalg.eig(self.sigma[:,:,i])
                    val = np.real(np.diag(val))
                    val[val < 1e-6] = 1e-6
                    self.sigma[:,:,i] = vec.dot(np.diag(val)).dot(vec.T)
                    self.prec[:,:,i] = vec.dot(np.diag(1.0/val)).dot(vec.T)
                    self.logdet[i,1] = np.sum(np.log(val))
                    """
                else: # Use quick and dirty regularization.
                    self.sigma[:,:,i] = 0.5*(self.sigma[:,:,i]+self.sigma[:,:,i].T) + 1e-6*np.eye(Do)

def test():
    from algorithm.cost.cost_utils import approx_equal
    np.random.seed(12345)
    gmm = GMM()
    data =np.random.randn(3, 10)
    gmm.update(data, 2)
    mu0, Phi, m, n0 = gmm.inference(data)
    assert approx_equal(mu0, np.array([ 0.49483644, 0.13111947, 0.38831296]))
    assert approx_equal(Phi, np.array([[ 0.64402048, 0.25949521,-0.18127717],
                                       [ 0.25949521, 1.23896766,-0.30703882],
                                       [-0.18127717,-0.30703882, 1.31296918]]))
    assert m == 1.0
    assert n0 == 0.5
