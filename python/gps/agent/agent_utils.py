import numpy as np
import scipy.ndimage as sp_ndimage


def generate_noise(T, dU, smooth=False, var=1.0, renorm=False):
    """
    Generate a T x dU gaussian-distributed noise vector.
    This will approximately have mean 0 and variance 1, ignoring smoothing.

    Args:
        T (int): # Timesteps
        dU (int): Dimension of actions
        smooth (bool, optional): Perform smoothing of noise.
        var (float, optional): If smooth=True, applies a gaussian filter with this variance.
        renorm (bool, optional): If smooth=True, renormalizes data to have variance 1 after smoothing.

    Sanity Check
    >>> np.random.seed(123)
    >>> generate_noise(5, 2)
    array([[-1.0856306 ,  0.99734545],
           [ 0.2829785 , -1.50629471],
           [-0.57860025,  1.65143654],
           [-2.42667924, -0.42891263],
           [ 1.26593626, -0.8667404 ]])
    >>> np.random.seed(123)
    >>> generate_noise(5, 2, smooth=True, var=0.5)
    array([[-0.93944619,  0.73034299],
           [ 0.04449717, -0.90269245],
           [-0.68326104,  1.09300178],
           [-1.8351787 , -0.25446477],
           [ 0.87139343, -0.81935331]])
    """
    noise = np.random.randn(T, dU)
    if smooth:
        # Smooth out the noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise/np.sqrt(variance)
    return noise
