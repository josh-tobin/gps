import numpy as np
import scipy as sp

def generate_noise(T, Du, smooth=False, smooth_sigma=1.0, smooth_renormalize=False):
    """
    Generate a T x Du noise vector.
    This will have mean 0 and variance 1, ignoring smoothing.


    """
    noise = np.random.randn((T, Du))
    if smooth:
        # Smooth out the noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(Du):
            noise[:, i] = sp.ndimage.filters.gaussian_filter(noise[:, i], smooth_sigma)
        if smooth_renormalize:
            variance = np.std(noise, axis=0)
            noise = noise/variance
    return noise