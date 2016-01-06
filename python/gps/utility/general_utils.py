import numpy as np


class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable namedtuple.
    """
    def __init__(self, vars):
        for var, val in vars.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape

    Args:
        value: Matrix to shape check
        expected_shape: A tuple or list of integers
        name (optional): An optional name to add to the exception message.
            Default empty string.
    Returns: None
    >>> a = np.zeros((4, 6))
    >>> check_shape(a, (4,6))
    >>> check_shape(a, (4,), name='a')
    Traceback (most recent call last):
    ...
    ValueError: Shape mismatch a: Expected (4,), got (4, 6)
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
            (name, str(expected_shape), str(value.shape)))


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


def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified condition,
    and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val \
            for var, val in hyperparams.items()}
