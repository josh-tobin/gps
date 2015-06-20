"""
Initializations for Linear-gaussian controllers
"""
import numpy as np
from algorithm.dynamics.dynamics_util import guess_dynamics_pos_vel


def init_lqr(x0, dX, dU, dt, T, gains, acc, init_stiffness, init_stiffness_vel, init_var, init_final_weight):
    """
    Return initial gains for a linear gaussian controller.
    This code runs, but has not been tested for correctness

    Some sanity checks
    >>> x0 = np.zeros(8)
    >>> ref, K, k, PSig, cholPSig, invPSig = init_lqr(x0, 8, 3, 0.1, 5, np.ones(3), np.ones(3), 1.0,1.0,1.0,1.0)
    >>> K.shape
    (5, 3, 8)
    """
    ref = np.hstack([np.tile(x0, [T, 1]), np.zeros((T, dU))])

    # Constants.
    ix = slice(dX)
    iu = slice(dX, dX + dU)

    # Set up simple linear model.
    fd, fc = guess_dynamics_pos_vel(gains, acc, dX, dU, dt)
    # Set up cost function.
    Cm = np.diag(np.hstack([init_stiffness*np.ones(dU),
                            init_stiffness*init_stiffness_vel*np.ones(dU),
                            np.zeros(dX-dU*2),
                            np.ones(dU)]))
    Cm = Cm / init_var
    cv = np.zeros(dX + dU)
    # Perform dynamic programming.
    K = np.zeros((dU, dX, T))
    k = np.zeros((dU, T))
    PSig = np.zeros((dU, dU, T))
    cholPSig = np.zeros((dU, dU, T))
    invPSig = np.zeros((dU, dU, T))
    Vxx = np.zeros((dX, dX))
    Vx = np.zeros(dX)
    for t in range(T-1, 0, -1):
        # Compute Q function at this step.
        if t == T:
            Cmt = init_final_weight * Cm
            cvt = init_final_weight * cv
        else:
            Cmt = Cm
            cvt = cv
        Qtt = Cmt + fd.T.dot(Vxx).dot(fd)
        Qt = cvt + fd.T.dot(Vx + Vxx.dot(fc))

        # Compute preceding value function.
        L = np.linalg.cholesky(Qtt[iu, iu])
        invPSig[:, :, t] = Qtt[iu, iu]
        PSig[:, :, t] = np.linalg.pinv(L).dot(np.linalg.pinv(L.T).dot(np.eye(dU)))
        cholPSig[:, :, t] = np.linalg.cholesky(PSig[:, :, t])
        K[:, :, t] = -np.linalg.pinv(L).dot(np.linalg.pinv(L.T).dot(Qtt[iu, ix]))
        k[:, t] = -np.linalg.pinv(L).dot(np.linalg.pinv(L.T).dot(Qt[iu]))
        Vxx = Qtt[ix, ix] + Qtt[ix, iu].dot(K[:, :, t])
        Vx = Qt[ix] + Qtt[ix, iu].dot(k[:, t])
        Vxx = 0.5 * (Vxx + Vxx.T)

    K = np.transpose(K, [2, 0, 1])
    k = k.T
    PSig = np.transpose(PSig, [2, 0, 1])
    cholPSig = np.transpose(cholPSig, [2, 0, 1])
    invPSig = np.transpose(invPSig, [2, 0, 1])
    return ref, K, k, PSig, cholPSig, invPSig


def init_pd(x0, dU, dQ, dX, T, init_stiffness, init_stiffness_vel, init_var, init_action_offset=None):
    """
    Return initial gains for a linear gaussian controller.
    This code runs, but has not been tested for correctness

    Returns:
        ref: T x dX+dU Reference trajectory + actions
        K: T x dU x dX linear controller matrix
        k: T x dU controller bias term
        PSig: T x dU x dU controller action covariance
        cholPSig: Cholesky decomposition of PSig
        invPSig: Inverse of PSig

    Some sanity checks
    >>> x0 = np.zeros(8)
    >>> ref, K, k, PSig, cholPSig, invPSig = init_pd(x0, 3, 3, 8, 5, 1.0, 1.0, 1.0)
    >>> K.shape
    (5, 3, 8)
    """
    ref = np.hstack([np.tile(x0, [T, 1]), np.zeros((T, dU))])

    # Choose initialization mode.
    Kp = 1.0
    Kv = init_stiffness_vel
    if dU < dQ:
        K = -init_stiffness * np.tile(
            [np.eye(dU) * Kp, np.zeros(dU, dQ - dU), np.eye(dU) * Kv, np.zeros((dU, dQ - dU))], [T, 1, 1])
    else:
        K = -init_stiffness * np.tile(np.hstack([np.eye(dU) * Kp, np.eye(dU) * Kv, np.zeros((dU, dX - dU * 2))]), [T, 1, 1])
    k = np.zeros((T, dU))
    if init_action_offset:
        ref[dX:, :] = np.tile(init_action_offset, [T, 1])
    PSig = init_var * np.tile(np.eye(dU), [T, 1, 1])
    cholPSig = np.sqrt(init_var) * np.tile(np.eye(dU), [T, 1, 1])
    invPSig = (1. / init_var) * np.tile(np.eye(dU), [T, 1, 1])

    return ref, K, k, PSig, cholPSig, invPSig