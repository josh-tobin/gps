"""
Initializations for Linear-gaussian controllers
"""
from copy import deepcopy
import numpy as np

from algorithm.dynamics.dynamics_util import guess_dynamics
from lin_gauss_policy import LinearGaussianPolicy
from config import init_lg
# TODO - put other arguments into a dictionary? include in hyperparams?


def init_lqr(hyperparams, x0, dX, dU, dt, T):
    """
    Return initial gains for a time-varying linear gaussian controller
    that tries to hold the initial position.

    Some sanity checks
    >>> x0 = np.zeros(8)
    >>> traj = init_lqr({}, x0, 8, 3, 0.1, 5)
    >>> traj.K.shape
    (5, 3, 8)
    """
    config = deepcopy(init_lg)
    config.update(hyperparams)
    #TODO: Use packing instead of assuming which indices are the joint angles.
    ref = np.hstack([np.tile(x0, [T, 1]), np.zeros((T, dU))])

    # Notation notes:
    # L = loss, Q = q-function (dX+dU dimensional), V = value function (dX dimensional), F = dynamics
    # Vectors are lower-case, Matrices are upper case
    # Derivatives: x = state, u = action, t = state+action (trajectory). All lowercase
    # The time index is denoted by _t after the above.
    # Ex. Ltt_t = Loss, 2nd derivative (w.r.t. trajectory), indexed by time t.

    # Constants.
    idx_x = slice(dX)  # Slices out state
    idx_u = slice(dX, dX + dU)  # Slices out actions

    if len(config['init_acc']) == 0:
        config['init_acc'] = np.ones(dU)

    if len(config['init_gains']) == 0:
        config['init_gains'] = np.ones(dU)

    # Set up simple linear dynamics model.
    Fd, fc = guess_dynamics(config['init_gains'], config['init_acc'], dX, dU, dt)

    # Setup a cost function based on stiffness.
    # Ltt = (dX+dU) by (dX+dU) - second derivative of loss with respect to trajectory at a single timestep
    Ltt = np.diag(np.hstack([config['init_stiffness']*np.ones(dU),
                            config['init_stiffness']*config['init_stiffness_vel']*np.ones(dU),
                            np.zeros(dX-dU*2),
                            np.ones(dU)]))
    Ltt = Ltt / config['init_var']
    lt = np.zeros(dX + dU)  # lt = (dX+dU) - first derivative of loss with respect to trajectory at a single timestep

    # Perform dynamic programming.
    K = np.zeros((T, dU, dX))  # Controller gains matrix
    k = np.zeros((T, dU))  # Controller bias term
    PSig = np.zeros((T, dU, dU))  # Covariance of noise
    cholPSig = np.zeros((T, dU, dU))  # Cholesky decomposition of covariance
    invPSig = np.zeros((T, dU, dU))  # Inverse of covariance
    Vxx = np.zeros((dX, dX))  # Vxx = ddV/dXdX. Second deriv of value function at some timestep.
    vx = np.zeros(dX)  # Vx = dV/dX. Derivative of value function at some timestep.
    for t in range(T-1, -1, -1):
        # Compute Q function at this step.
        if t == (T-1):
            Ltt_t = config['init_final_weight'] * Ltt
            lt_t = config['init_final_weight'] * lt
        else:
            Ltt_t = Ltt
            lt_t = lt
        # Qtt = (dX+dU) by (dX+dU) 2nd Derivative of Q-function with respect to trajectory (dX+dU)
        Qtt = Ltt_t + Fd.T.dot(Vxx).dot(Fd)
        # Qt = (dX+dU) 1st Derivative of Q-function with respect to trajectory (dX+dU)
        qt = lt_t + Fd.T.dot(vx + Vxx.dot(fc))

        # Compute preceding value function.
        L = np.linalg.cholesky(Qtt[idx_u, idx_u])
        invPSig[t, :, :] = Qtt[idx_u, idx_u]
        PSig[t, :, :] = np.linalg.inv(L).dot(np.linalg.inv(L.T).dot(np.eye(dU)))
        cholPSig[t, :, :] = np.linalg.cholesky(PSig[t, :, :])
        K[t, :, :] = -np.linalg.inv(L).dot(np.linalg.inv(L.T).dot(Qtt[idx_u, idx_x]))
        k[t, :] = -np.linalg.inv(L).dot(np.linalg.inv(L.T).dot(qt[idx_u]))
        Vxx = Qtt[idx_x, idx_x] + Qtt[idx_x, idx_u].dot(K[t, :, :])
        vx = qt[idx_x] + Qtt[idx_x, idx_u].dot(k[t, :])
        Vxx = 0.5 * (Vxx + Vxx.T)

    return LinearGaussianPolicy(K, k, ref, PSig, cholPSig, invPSig)


def init_pd(hyperparams, x0, dU, dQ, dX, T):
    """
    Return initial gains for a time-varying linear gaussian controller that
    tries to hold the initial position.

    Returns:
        ref: T x dX+dU Reference trajectory
        K: T x dU x dX linear controller gains matrix
        k: T x dU controller bias term
        PSig: T x dU x dU controller action covariance
        cholPSig: Cholesky decomposition of PSig
        invPSig: Inverse of PSig

    Some sanity checks
    >>> x0 = np.zeros(8)
    >>> traj = init_pd({}, x0, 3, 3, 8, 5)
    >>> traj.K.shape
    (5, 3, 8)
    """
    config = deepcopy(init_lg)
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

    return LinearGaussianPolicy(K, k, ref, PSig, cholPSig, invPSig)
