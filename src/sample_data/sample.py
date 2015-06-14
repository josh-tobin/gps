import numpy as np
from gps_sample_types import *

class Sample(object):
    """Class that handles the representation of a trajectory and stores a single trajectory

    Note: must be serializable for easy saving - no C++ object references!
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

        self.T = self._hyperparams['T']
        self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.dPhi = self._hyperparams['dPhi']

        # list of numpy arrays containing the sample data from various sensors
        self._data = []

        # To be populated in by the C++ object, maps sensor name to index
        self._sensor_idx = {} # Actually now just DataType object

        self._X = np.empty([self.T, self.dX])
        self._X.fill(np.nan)
        self._U = np.empty([self.T, self.dU])
        self._U.fill(np.nan)
        self._phi = np.empty([self.T, self.dPhi])
        self._phi.fill(np.nan)

    def set(self, sensor_name, sensor_data):
        """Set trajectory data for a particular sensor"""
        self._data[self._sensor_idx[sensor_name]] = sensor_data

    def get(self, sensor_name):
        """Get trajectory data for a particular sensor"""
        return self._data[self._sensor_idx[sensor_name]]

    def get_X(self):
        """Get the state. Put it together if not already precomputed."""
        if np.any(np.isnan(self._X)):
            raise NotImplementedError("TODO - Compute _X by calling C++ code");
        return self._X

    def get_U(self):
        """Get the action. Put it together if not already precomputed."""
        if np.any(np.isnan(self._U)):
            raise NotImplementedError("TODO - Compute _U by calling C++ code");
        return self._U

    def get_phi(self):
        """Get the feature representation. Put it together if not already precomputed."""
        if np.any(np.isnan(self._phi)):
            raise NotImplementedError("TODO - Compute _phi by calling C++ code");
        return self._phi
