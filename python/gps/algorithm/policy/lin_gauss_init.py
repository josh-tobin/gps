"""
Initializations for Linear-gaussian controllers
"""
from algorithm.dynamics.dynamics_util import guess_dynamics
from config import init_lg

import numpy as np

# TODO - put other arguments into a dictionary? include in hyperparams?
def init_lqr(hyperparams, x0, dX, dU, dt, T):
    """
    Return initial gains for a time-varying linear gaussian controller.

    *** This code runs, but returns incorrect results.

    Some sanity checks
    >>> x0 = np.zeros(8)
    >>> ref, K, k, PSig, cholPSig, invPSig = init_lqr(x0, 8, 3, 0.1, 5, np.ones(3), np.ones(3), 1.0,1.0,1.0,1.0)
    >>> K.shape
    (5, 3, 8)
    """
    config = init_lg.deepcopy()
    config.update(hyperparams)
    #TODO: Use packing instead of assuming which indices are the joint angles.
    #TODO: Comment on variables.
    ref = np.hstack([np.tile(x0, [T, 1]), np.zeros((T, dU))])

    # Constants.
    ix = slice(dX)
    iu = slice(dX, dX + dU)

    if len(config['init_acc']) == 0:
        config['init_acc'] = np.ones(dU)

    if len(config['init_gains']) == 0:
        config['init_gains'] = np.ones(dU)

    # Set up simple linear dynamics model.
    fd, fc = guess_dynamics(gains, config['init_acc'], dX, dU, dt)
    # Set up cost function.
    Cm = np.diag(np.hstack([config['init_stiffness']*np.ones(dU),
                            config['init_stiffness']*config['init_stiffness_vel']*np.ones(dU),
                            np.zeros(dX-dU*2),
                            np.ones(dU)]))
    Cm = Cm / config['init_var']
    cv = np.zeros(dX + dU)  # Derivative of cost
    # Perform dynamic programming.
    K = np.zeros((dU, dX, T))
    k = np.zeros((dU, T))
    PSig = np.zeros((dU, dU, T))
    cholPSig = np.zeros((dU, dU, T))
    invPSig = np.zeros((dU, dU, T))
    Vxx = np.zeros((dX, dX))  # Vxx = ddV/dXdX. Second deriv of value function at some timestep.
    Vx = np.zeros(dX)  # Vx = dV/dX. Derivative of value function at some timestep.
    for t in range(T-1, -1, -1):
        # Compute Q function at this step.
        if t == T:
            Cmt = config['init_final_weight'] * Cm
            cvt = config['init_final_weight'] * cv
        else:
            Cmt = Cm
            cvt = cv
        Qtt = Cmt + fd.T.dot(Vxx).dot(fd)
        Qt = cvt + fd.T.dot(Vx + Vxx.dot(fc))

        #TODO: There is incorrectness here (need to debug)
        # Compute preceding value function.
        L = np.linalg.cholesky(Qtt[iu, iu])
        invPSig[:, :, t] = Qtt[iu, iu]
        PSig[:, :, t] = np.linalg.inv(L).dot(np.linalg.inv(L.T).dot(np.eye(dU)))
        cholPSig[:, :, t] = np.linalg.cholesky(PSig[:, :, t])
        K[:, :, t] = -np.linalg.inv(L).dot(np.linalg.inv(L.T).dot(Qtt[iu, ix]))
        k[:, t] = -np.linalg.inv(L).dot(np.linalg.inv(L.T).dot(Qt[iu]))
        Vxx = Qtt[ix, ix] + Qtt[ix, iu].dot(K[:, :, t])
        Vx = Qt[ix] + Qtt[ix, iu].dot(k[:, t])
        Vxx = 0.5 * (Vxx + Vxx.T)

    #TODO: Remove tranposes once code is verified to be correct.
    K = np.transpose(K, [2, 0, 1])
    k = k.T
    PSig = np.transpose(PSig, [2, 0, 1])
    cholPSig = np.transpose(cholPSig, [2, 0, 1])
    invPSig = np.transpose(invPSig, [2, 0, 1])
    return ref, K, k, PSig, cholPSig, invPSig


def init_pd(hyperparams, x0, dU, dQ, dX, T):
    """
    Return initial gains for a time-varying linear gaussian controller.

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
    config = init_lg.deepcopy()
    config.update(hyperparams)
    ref = np.hstack([np.tile(x0, [T, 1]), np.zeros((T, dU))])

    # Choose initialization mode.
    Kp = 1.0
    Kv = config['init_stiffness_vel']
    if dU < dQ:
        K = -config['init_stiffness'] * np.tile(
            [np.eye(dU) * Kp, np.zeros(dU, dQ - dU), np.eye(dU) * Kv, np.zeros((dU, dQ - dU))], [T, 1, 1])
    else:
        K = -config['init_stiffness'] * np.tile(np.hstack([np.eye(dU) * Kp, np.eye(dU) * Kv, np.zeros((dU, dX - dU * 2))]), [T, 1, 1])
    k = np.zeros((T, dU))
    if config['init_action_offset']:
        ref[dX:, :] = np.tile(config['init_action_offset'], [T, 1])
    PSig = config['init_var'] * np.tile(np.eye(dU), [T, 1, 1])
    cholPSig = np.sqrt(config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])
    invPSig = (1. / config['init_var']) * np.tile(np.eye(dU), [T, 1, 1])

    return ref, K, k, PSig, cholPSig, invPSig
