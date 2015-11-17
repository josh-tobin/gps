import numpy as np


def estimate_moments(X, mu, covar):
    """
    Estimate the moments for a given linearized policy.
    """
    N, T, dX = X.shape
    dU = mu.shape[-1]
    if len(covar.shape) == 3:
        covar = np.tile(covar, [N, 1, 1, 1])
    Xmu = np.concatenate([X, mu], axis=2)
    ev = np.mean(Xmu, axis=0)
    em = np.zeros((N, T, dX+dU, dX+dU))
    pad1 = np.zeros((dX,dX+dU))
    pad2 = np.zeros((dU,dX))
    for n in range(N):
        for t in range(T):
            covar_pad = np.vstack([pad1, np.hstack([pad2, covar[n,t,:,:]])])
            em[n,t,:,:] = np.outer(Xmu[n,t,:], Xmu[n,t,:]) + covar_pad
    return ev, em


def gauss_fit_joint_prior(pts, mu0, Phi, m, n0, dwts, dX, dU, sig_reg):
    """
    Perform Gaussian fit to data with a prior.
    """
    # Build weights matrix.
    D = np.diag(dwts)
    # Compute empirical mean and covariance.
    mun = np.sum((pts.T * dwts).T, axis=0)
    diff = pts - mun
    empsig = diff.T.dot(D).dot(diff)
    empsig = 0.5 * (empsig + empsig.T)
    # MAP estimate of joint distribution.
    N = dwts.shape[0]
    mu = mun
    sigma = (N * empsig + Phi + ((N*m)/(N+m)) * np.outer(mun-mu0, mun-mu0)) / (N + n0)
    sigma = 0.5 * (sigma + sigma.T)
    # Add sigma regularization.
    sigma += sig_reg
    # Conditioning to get dynamics.
    fd = np.linalg.solve(sigma[:dX,:dX], sigma[:dX,dX:dX+dU]).T
    fc = mu[dX:dX+dU] - fd.dot(mu[:dX])
    dynsig = sigma[dX:dX+dU,dX:dX+dU] - fd.dot(sigma[:dX,:dX]).dot(fd.T)
    dynsig = 0.5 * (dynsig + dynsig.T)
    return fd, fc, dynsig
