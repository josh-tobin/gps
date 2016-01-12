import numpy as np
from gps.utility import finite_differences, approx_equal


def finite_differences_cost_test(cost, x, u, obs, sample_meta, epsilon=1e-5, threshold=1e-5):
    """
    Finite-differences cost function checker. ***Broken for now.***
    """
    #TODO: Lots of repeated code blocks here - can make this a bit cleaner.
    #TODO: This needs significant rewriting, perhaps we should just remove this.

    l, lx, lu, lxx, luu, lux = cost.eval(x, u, obs, sample_meta)
    T, Dx = x.shape
    _, Du = u.shape

    lx_test = np.zeros_like(lx)
    for t in range(T):
        loss_func = lambda x_test: cost.eval(x_test, u[t,:], obs[t,:], [sample_meta[t]])[0]
        x_input = x[t,:]
        lx_test[t] = finite_differences(loss_func, x_input, func_output_shape=(1, 1),
                epsilon=epsilon)[0,:,0,0]
    for idx, _ in np.ndenumerate(lx):
        equal = approx_equal(lx_test[idx], lx[idx], threshold=threshold)
        if not equal:
            raise ValueError("lx not equal: %f vs %f" % (lx_test[idx], lx[idx]))

    lu_test = np.zeros_like(lu)
    for t in range(T):
        loss_func = lambda u_test: cost.eval(x[t,:], u_test, obs[t,:], [sample_meta[t]])[0]
        u_input = u[t,:]
        lu_test[t] = finite_differences(loss_func, u_input, func_output_shape=(1, 1),
                epsilon=epsilon)[0,:,0,0]
    for idx, _ in np.ndenumerate(lu):
        equal = approx_equal(lu_test[idx], lu[idx], threshold=threshold)
        if not equal:
            raise ValueError("lu not equal")

    lxx_test = np.zeros_like(lxx)
    for t in range(T):
        loss_func = lambda x_test: cost.eval(x_test, u[t,:], obs[t,:], [sample_meta[t]])[1]
        x_input = x[t,:]
        lxx_test[t] = finite_differences(loss_func, x_input, func_output_shape=(1, Dx),
                epsilon=epsilon)[0,:,0,:].T
    for idx, _ in np.ndenumerate(lxx):
        equal = approx_equal(lxx_test[idx], lxx[idx], threshold=threshold)
        if not equal:
            raise ValueError("lxx not equal")

    luu_test = np.zeros_like(luu)
    for t in range(T):
        loss_func = lambda u_test: cost.eval(x[t,:], u_test, obs[t,:], [sample_meta[t]])[2]
        u_input = u[t,:]
        luu_test[t] = finite_differences(loss_func, u_input, func_output_shape=(1, Du),
                epsilon=epsilon)[0,:,0,:].T
    for idx, _ in np.ndenumerate(luu):
        equal = approx_equal(luu_test[idx], luu[idx], threshold=threshold)
        if not equal:
            raise ValueError("luu not equal")

    lux_test = np.zeros_like(lux)
    for t in range(T):
        loss_func = lambda x_test: cost.eval(x_test, u[t,:], obs[t,:], [sample_meta[t]])[2]
        x_input = x[t,:]
        lux_test[t] = finite_differences(loss_func, x_input, func_output_shape=(1, Du),
                epsilon=epsilon)[0,:,0,:].T
    for idx, _ in np.ndenumerate(lux):
        equal = approx_equal(lux_test[idx], lux[idx], threshold=threshold)
        if not equal:
            raise ValueError("lux not equal")

    return True


RAMP_CONSTANT = 1
RAMP_LINEAR = 2
RAMP_QUADRATIC = 3
RAMP_FINAL_ONLY = 4


def get_ramp_multiplier(ramp_option, T, wp_final_multiplier=1.0):
    """
    Returns a time-varying multiplier.
    """
    if ramp_option == RAMP_CONSTANT:
        wpm = np.ones(T)
    elif ramp_option == RAMP_LINEAR:
        wpm = (np.arange(T, dtype=np.float32) + 1) / T
    elif ramp_option == RAMP_QUADRATIC:
        wpm = ((np.arange(T, dtype=np.float32) + 1) / T) ** 2
    elif ramp_option == RAMP_FINAL_ONLY:
        wpm = np.zeros(T)
        wpm[T-1] = 1.0
    else:
        raise ValueError('Unknown cost ramp requested!')
    wpm[-1] *= wp_final_multiplier
    return wpm


def evall1l2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (l1 * sqrt(alpha + d^2))
    Args:
        wp: T x D matrix containing weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + np.sqrt(alpha + np.sum(dscl ** 2, axis=1)) * l1

    # First order derivative terms.
    d1 = dscl * l2 + (dscls / np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1)
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims(np.sqrt(alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1)
    d2 = l1 * \
            ((np.expand_dims(np.eye(wp.shape[1]), axis=0) * (np.expand_dims(wp ** 2, axis=1) / psq)) \
            - ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=2)) / psq ** 3))
    d2 += l2 * (np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1]))

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum((Jd_expand_1 * Jd_expand_2) * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0,2,1])

    return l, lx, lxx


def evallogl2term(wp, d, Jd, Jdd, l1, l2, alpha):
    """
    Evaluate and compute derivatives for combined l1/l2 norm penalty.
    loss = (0.5 * l2 * d^2) + (0.5 * l1 * log(alpha + d^2))
    Args:
        wp: T x D matrix containing weights for each dimension and time step.
        d: T x D states to evaluate norm on.
        Jd: T x D x Dx Jacobian - derivative of d with respect to state.
        Jdd: T x D x Dx x Dx Jacobian - 2nd derivative of d with respect to state.
        l1: l1 loss weight.
        l2: l2 loss weight.
        alpha: Constant added in square root.
    """
    # Get trajectory length.
    T, _ = d.shape

    # Compute scaled quantities.
    sqrtwp = np.sqrt(wp)
    dsclsq = d * sqrtwp
    dscl = d * wp
    dscls = d * (wp ** 2)

    # Compute total cost.
    l = 0.5 * np.sum(dsclsq ** 2, axis=1) * l2 + 0.5*np.log(alpha + np.sum(dscl ** 2, axis=1)) * l1 
    # First order derivative terms.
    d1 = dscl * l2 + (dscls / (alpha + np.sum(dscl ** 2, axis=1, keepdims=True)) * l1)
    lx = np.sum(Jd * np.expand_dims(d1, axis=2), axis=1)

    # Second order terms.
    psq = np.expand_dims((alpha + np.sum(dscl ** 2, axis=1, keepdims=True)), axis=1)
    #TODO: need * 2.0 somewhere in following line, or * 0.0 which is wrong but better.
    d2 = l1 * \
            ((np.expand_dims(np.eye(wp.shape[1]), axis=0) * (np.expand_dims(wp ** 2, axis=1) / psq)) \
            - ((np.expand_dims(dscls, axis=1) * np.expand_dims(dscls, axis=2)) / psq ** 2))
    d2 += l2 * (np.expand_dims(wp, axis=2) * np.tile(np.eye(wp.shape[1]), [T, 1, 1]))

    d1_expand = np.expand_dims(np.expand_dims(d1, axis=-1), axis=-1)
    sec = np.sum(d1_expand * Jdd, axis=1)

    Jd_expand_1 = np.expand_dims(np.expand_dims(Jd, axis=2), axis=4)
    Jd_expand_2 = np.expand_dims(np.expand_dims(Jd, axis=1), axis=3)
    d2_expand = np.expand_dims(np.expand_dims(d2, axis=-1), axis=-1)
    lxx = np.sum(np.sum((Jd_expand_1 * Jd_expand_2) * d2_expand, axis=1), axis=1)

    lxx += 0.5 * sec + 0.5 * np.transpose(sec, [0,2,1])

    return l, lx, lxx
