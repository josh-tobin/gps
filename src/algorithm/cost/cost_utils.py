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

    >>> approx_equal(2.0,2.0000001)
    True
    >>> approx_equal(np.array([2.0, 1.0]), np.array([2.0, 1.0]))
    True
    >>> approx_equal(np.array([2.0, 1.0]), np.array([2.0, 1.001]))
    False
    """
    return np.all(np.abs(a - b) < threshold)


def finite_differences_cost_test(cost, x, u, obs, sample_meta, epsilon=1e-5, threshold=1e-5):
    """
    Finite-differences cost function checker
    TODO: More docstrings
    """
    # TODO: Have some tests for the tester
    # TODO: Lots of repeated code blocks here - can make this a bit cleaner

    l, lx, lu, lxx, luu, lux = cost.eval(x, u, obs, sample_meta)
    T, Dx = x.shape
    _, Du = u.shape

    #Check lx
    lx_test = np.zeros_like(lx)
    for t in range(T):
        loss_func = lambda x_test:  cost.eval(x_test, u[[t], :], obs[[t], :], [sample_meta[t]])[0]
        x_input = x[[t], :]
        lx_test[t] = finite_differences(loss_func, x_input, func_output_shape=(1, 1), epsilon=epsilon)[0, :, 0, 0]
    for idx, _ in np.ndenumerate(lx):
        equal = approx_equal(lx_test[idx], lx[idx], threshold=threshold)
        if not equal:
            raise ValueError("lx not equal")

    #Check lu
    lu_test = np.zeros_like(lu)
    for t in range(T):
        loss_func = lambda u_test:  cost.eval(x[[t], :], u_test, obs[[t], :], [sample_meta[t]])[0]
        u_input = u[[t], :]
        lu_test[t] = finite_differences(loss_func, u_input, func_output_shape=(1, 1), epsilon=epsilon)[0, :, 0, 0]
    for idx, _ in np.ndenumerate(lu):
        equal = approx_equal(lu_test[idx], lu[idx], threshold=threshold)
        if not equal:
            raise ValueError("lu not equal")

    #Check lxx
    lxx_test = np.zeros_like(lxx)
    for t in range(T):
        loss_func = lambda x_test:  cost.eval(x_test, u[[t], :], obs[[t], :], [sample_meta[t]])[1]
        x_input = x[[t], :]
        lxx_test[t] = finite_differences(loss_func, x_input, func_output_shape=(1, Dx), epsilon=epsilon)[0, :, 0, :].T
    for idx, _ in np.ndenumerate(lxx):
        equal = approx_equal(lxx_test[idx], lxx[idx], threshold=threshold)
        if not equal:
            raise ValueError("lxx not equal")

    #Check luu
    luu_test = np.zeros_like(luu)
    for t in range(T):
        loss_func = lambda u_test:  cost.eval(x[[t], :], u_test, obs[[t], :], [sample_meta[t]])[2]
        u_input = u[[t], :]
        luu_test[t] = finite_differences(loss_func, u_input, func_output_shape=(1, Du), epsilon=epsilon)[0, :, 0, :].T
    for idx, _ in np.ndenumerate(luu):
        equal = approx_equal(luu_test[idx], luu[idx], threshold=threshold)
        if not equal:
            raise ValueError("luu not equal")

    #Check lux
    lux_test = np.zeros_like(lux)
    for t in range(T):
        loss_func = lambda x_test:  cost.eval(x_test, u[[t], :], obs[[t], :], [sample_meta[t]])[2]
        x_input = x[[t], :]
        lux_test[t] = finite_differences(loss_func, x_input, func_output_shape=(1, Du), epsilon=epsilon)[0, :, 0, :].T
    for idx, _ in np.ndenumerate(lux):
        equal = approx_equal(lux_test[idx], lux[idx], threshold=threshold)
        if not equal:
            raise ValueError("lux not equal")


def finite_differences(func, inputs, func_output_shape=(), epsilon=1e-5):
    """
    Computes gradients via finite differences.

                   func(x+epsilon)-func(x-epsilon)
    derivative =      ------------------------
                            2*epsilon

    Args:
        func: Function to compute gradient of. Inputs and outputs can be arbitrary dimension.
        inputs (float vector/matrix): Vector value to compute gradient at
        func_output_shape (int tuple, optional): Shape of the output of func. Default is empty-tuple,
            which works for scalar-valued functions.
        epsilon (float, optional): Difference to use for computing gradient.

    Returns:
        Gradient vector of each dimension of func with respect to each dimension of input.
        Will be of shape (inputs_dim X func_output_shape)

    Doctests/Example usages:
    >>> import numpy as np

    #Test vector-shaped gradient
    >>> func = lambda x: x.dot(x)
    >>> finite_differences(func, np.array([1.0, 4.0, 9.0]))
    array([  2.,   8.,  18.])

    #Test matrix-shaped gradient
    >>> func = lambda x: np.sum(x)
    >>> finite_differences(func, np.array([[1.0, 2.0], [3.0, 4.0]]))
    array([[ 1.,  1.],
           [ 1.,  1.]])

    #Test multi-dim objective function. 2nd derivative of x.dot(x)
    >>> func = lambda x: 2*x
    >>> finite_differences(func, np.array([1.0, 2.0]), func_output_shape=(2,))
    array([[ 2.,  0.],
           [ 0.,  2.]])
    """
    gradient = np.zeros(inputs.shape+func_output_shape)
    for idx, _ in np.ndenumerate(inputs):
        test_input = np.copy(inputs)
        test_input[idx] += epsilon
        obj_d1 = func(test_input)
        assert obj_d1.shape == func_output_shape
        test_input = np.copy(inputs)
        test_input[idx] -= epsilon
        obj_d2 = func(test_input)
        assert obj_d2.shape == func_output_shape
        diff = (obj_d1-obj_d2) / (2*epsilon)
        gradient[idx] += diff
    return gradient


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
    >>> approx_equal(np.sum(lx),88150.426292309843)
    True
    >>> approx_equal(np.sum(lxx),6409790.2535816655, threshold=1e-2)
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


def evallogl2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """ Evaluate and compute derivatives for combined l1/l2 norm penalty. """
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
        + 0.5*np.log(alpha + np.sum(dscl ** 2, axis=0, keepdims=True)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / (alpha + np.sum(dscl ** 2, axis=0, keepdims=True)) * l1)
    lx = np.transpose(np.sum(Jd * np.expand_dims(d1, axis=0), axis=1, keepdims=True), [0, 2, 1])
    assert lx.shape[2] == 1
    lx = lx[:, :, 0]

    # Second order terms.
    psq = np.expand_dims((alpha + np.sum(dscl ** 2, axis=0, keepdims=True)), axis=0)
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
