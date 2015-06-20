import numpy as np


def guess_dynamics_jt(dX, dU, dP, dt):
    """
    Jacobian-transpose control.
    This code runs, but has not been tested for correctness

    Args:
        fd: Transition matrix
        fc: Bias vector
        dX:
        dU:
        dP: Dimension of points?
        dt: Timestep

    Some sanity checks
    >>> fd, fc = guess_dynamics_jt(9, 3, 1, 0.1)
    >>> fd.shape
    (9, 12)
    >>> fc.shape
    (9,)
    """
    fd = np.zeros((dX, dX + dU))
    fc = np.zeros(dX)
    mass = 1.0  # Some reasonable setting for in-hand mass for Jacobian-transpose control.
    hand_gains = (1 / mass) * np.ones(3)
    for i in range(dP):
        pidx_start = dX - 6 * dP + (i - 1) * 3
        pidx_end = dX - 6 * dP + (i - 1) * 3 + 3
        pidx = slice(pidx_start, pidx_end)
        vidx_start = dX - 3 * dP + (i - 1) * 3
        vidx_end = dX - 3 * dP + (i - 1) * 3 + 3
        vidx = slice(vidx_start, vidx_end)

        fd[pidx, pidx] = np.eye(3)
        fd[pidx, vidx] = dt * np.eye(3)
        fd[pidx, dX:dX+3] = (dt ** 2) * np.diag(hand_gains)
        fd[vidx, vidx] = np.eye(3)
        fd[vidx, dX:dX+3] = dt * np.diag(hand_gains)

    return fd, fc


def guess_dynamics_pos_vel(gains, acc, dX, dU, dt):
    """
    Initial guess at the model using position-velocity assumption.
    This code runs, but has not been tested for correctness

    Args:
        gains: dU dimensional joint gains
        acc: dU dimensional joint acceleration
        dX:
        dU:
        dt: Timestep

    Returns:
        fd: dX by dX+dU transition matrix
        fc: dX bias

    Some sanity checks
    >>> fd, fc = guess_dynamics_pos_vel(np.ones(3), np.ones(3), 9, 3, 0.1)
    >>> fd.shape
    (9, 12)
    >>> fc.shape
    (9,)
    """
    fd = np.vstack([
                    np.hstack([np.eye(dU), dt * np.eye(dU), np.zeros((dU, dX - dU * 2)), dt ** 2 * np.diag(gains)]),
                    np.hstack([np.zeros((dU, dU)), np.eye(dU), np.zeros((dU, dX - dU * 2)), dt * np.diag(gains)]),
                    np.zeros((dX - dU * 2, dX + dU))
                   ])
    fc = np.hstack([acc * dt ** 2, acc * dt, np.zeros((dX - dU * 2))])
    return fd, fc