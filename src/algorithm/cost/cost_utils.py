import numpy as np


def approx_equal(a, b, threshold=1e-5):
    """
    Return whether two numbers are equal within an absolute threshold

    Args:
        a (float):
        b (float):
        threshold (float, optional): Comparison threshold. Default 1e-5

    Returns:
        True if a and b are equal within threshold. 
    """
    return np.abs(a - b) < threshold


def finite_differences_test(cost, x, u, obs, sample_meta, epsilon=1e-5, threshold=1e-5):
    """
    Placeholder for a finite-differences cost function checker
    """
    raise NotImplementedError()


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.

    Args:
        wp: 
            TxD matrix containing weights for each dimension and timestep
        d:
            TxD
        Jd:
            TxDxD
        Jdd:
            TxDxDxD
        l1: l1 loss weight
        l2: l2 loss weight
        alpha:


    # Perform a quick sanity check
    >>> import numpy as np
    >>> wp = np.ones((100,39))
    >>> d = np.ones((100,39))
    >>> Jd = np.ones((100,39,39))
    >>> Jdd = np.ones((100,39,39,39))
    >>> l1 = 0.5
    >>> l2 = 0.5
    >>> alpha = 0.5
    >>> l,lx,lxx = evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha)
    >>> approx_equal(np.sum(l),1289.2451272494118)
    True
    >>> approx_equal(np.sum(np.sum(lx)),88150.426292309843)
    True
    >>> approx_equal(np.sum(np.sum(np.sum(lxx))),6409790.2535816655)
    True
    """
    # Evaluate a combined L1/L2 norm penalty.
    # TODO: Don't transpose everything in the beginning to match the matlab code
    d = d.T
    wp = wp.T
    Jd = np.transpose(Jd, [1, 2, 0])
    Jdd = np.transpose(Jdd, [1, 2, 3, 0])

    # Get trajectory length.
    _, T = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=0, keepdims=True) * l2 \
        + np.sqrt(alpha + np.sum(dscl ** 2, axis=0, keepdims=True)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=0, keepdims=True)) * l1)
    lx = np.transpose(np.sum(Jd * np.expand_dims(d1, axis=0), axis=1, keepdims=True), [0, 2, 1])
    assert lx.shape[2] == 1
    lx = lx[:, :, 0]

    # Second order terms.
    psq = np.expand_dims(np.sqrt(alpha + np.sum(dscl ** 2, axis=0, keepdims=True)), axis=0)
    d2 = l1 * ((np.expand_dims(np.eye(wp.shape[0]), axis=2) * (np.expand_dims(wp ** 2, axis=1) / psq)) -
               ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=0)) / psq ** 3))
    d2 += l2 * (np.expand_dims(wp, axis=1) * np.transpose(np.tile(np.eye(wp.shape[0]), [T, 1, 1]), [1, 2, 0]))

    d1_expand = np.expand_dims(np.expand_dims(d1.T, axis=0), axis=0)
    sec = np.sum(d1_expand * np.transpose(Jdd, [0, 1, 3, 2]), axis=3)

    Jd_expand_1 = np.expand_dims(np.expand_dims(np.transpose(Jd, [0, 2, 1]), axis=1), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(np.transpose(Jd, [0, 2, 1]), axis=2), axis=0)
    d2_expand = np.expand_dims(np.expand_dims(np.transpose(d2, [2, 0, 1]), axis=0), axis=0)
    lxx = np.sum(np.sum((Jd_expand_1 * Jd_expand_2) * d2_expand, axis=3, keepdims=True), axis=4, keepdims=True)
    assert lxx.shape[3:5] == (1, 1)
    lxx = lxx[:, :, :, 0, 0]

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [1, 0, 2])

    return l.T, lx.T, np.transpose(lxx, [2, 0, 1])
