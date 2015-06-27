import numpy as np


def guess_dynamics(gains, acc, dX, dU, dt):
    """
    Initial guess at the model using position-velocity assumption.

    *** This code assumes joint positions occupy the first dU state indices and
    joint velocities occupy the next dU.

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
    >>> Fd, fc = guess_dynamics(np.ones(3), np.ones(3), 9, 3, 0.1)
    >>> Fd.shape
    (9, 12)
    >>> fc.shape
    (9,)
    """
    #TODO: Use packing instead of assuming which indices are the joint angles.
    Fd = np.vstack([
                    np.hstack([np.eye(dU), dt * np.eye(dU), np.zeros((dU, dX - dU * 2)), dt ** 2 * np.diag(gains)]),
                    np.hstack([np.zeros((dU, dU)), np.eye(dU), np.zeros((dU, dX - dU * 2)), dt * np.diag(gains)]),
                    np.zeros((dX - dU * 2, dX + dU))
                   ])
    fc = np.hstack([acc * dt ** 2, acc * dt, np.zeros((dX - dU * 2))])
    return Fd, fc