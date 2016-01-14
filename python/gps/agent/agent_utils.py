import numpy as np
import scipy.ndimage as sp_ndimage


def generate_noise(T, dU, smooth=False, var=1.0, renorm=False):
    """
    Generate a T x dU gaussian-distributed noise vector. This will approximately have mean 0 and
    variance 1, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this variance.
        renorm : If smooth=True, renormalizes data to have variance 1 after smoothing.
    """
    noise = np.random.randn(T, dU)
    if smooth:
        # Smooth noise. This violates the controller assumption, but might produce smoother motions.
        for i in range(dU):
            noise[:,i] = sp_ndimage.filters.gaussian_filter(noise[:,i], var)
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise / np.sqrt(variance)
    return noise
