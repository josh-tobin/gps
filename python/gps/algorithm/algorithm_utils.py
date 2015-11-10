import numpy as np


def estimate_moments(X, mu, covar):
    """
    Estimate the moments for a given linearized policy.
    """
    N, T, dX = X.shape
    dU = mu.shape[-1]
    if covar.shape[0] == 1:
        covar = np.tile(covar, [N, 1, 1, 1])
    Xmu = np.concatenate([X, mu], axis=2)
    ev = np.mean(Xmu, axis=0)
    em = np.zeros((N, T, dX+dU, dX+dU))
    pad1 = np.zeros((dX,dX+dU))
    pad2 = np.zeros((dU,dX))
    for n in range(N):
        for t in range(T):
            covar_pad = np.hstack([pad1, np.vstack([pad2, covar[n,t,:,:]])])
            em[n,t,:,:] = np.outer(Xmu[n,t,:], Xmu[n,t,:]) + covar_pad
    return ev, em
