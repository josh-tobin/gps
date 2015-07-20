import numpy as np
from gps_sample_types import Action


class Sample(object):
    """Class that handles the representation of a trajectory and stores a single trajectory

    Note: must be serializable for easy saving - no C++ object references!
    """

    def __init__(self, sample_data):
        self.sample_data = sample_data

        self.T = sample_data.T
        self.dX = sample_data.dX
        self.dU = sample_data.dU
        self.dO = sample_data.dO

        # Dictionary containing the sample data from various sensors
        self._data = {}

        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)

    def set(self, sensor_name, sensor_data, t=None):
        """Set trajectory data for a particular sensor"""
        if t is None:
            self._data[sensor_name] = sensor_data
            self._X.fill(np.nan)  # Invalidate existing X
            self._obs.fill(np.nan)  # Invalidate existing Obs
        else:
            if sensor_name not in self._data:
                self._data[sensor_name] = np.empty((self.T,) + sensor_data.shape)
                self._data[sensor_name].fill(np.nan)
            self._data[sensor_name][t,:] = sensor_data
            self._X[t,:].fill(np.nan)
            self._obs[t,:].fill(np.nan)

    def get(self, sensor_name, t=None):
        """Get trajectory data for a particular sensor"""
        return self._data[sensor_name] if t is None else self._data[sensor_name][t,:]

    def get_X(self, t=None):
        """Get the state. Put it together if not already precomputed."""
        X = self._X if t is None else self._X[t,:]
        if np.any(np.isnan(X)):
            for data_type in self._data:
                if data_type not in self.sample_data.x_data_types:
                    continue
                data = self._data[data_type] if t is None else self._data[data_type][t,:]
                self.sample_data.pack_data_x(X, data, data_types=[data_type])
        return X

    def get_U(self, t=None):
        """Get the action. Put it together if not already precomputed."""
        return self._data[Action] if t is None else self._data[Action][t,:]

    def get_obs(self, t=None):
        """Get the feature representation. Put it together if not already precomputed."""
        obs = self._obs if t is None else self._obs[t,:]
        if np.any(np.isnan(obs)):
            for data_type in self._data:
                if data_type not in self.sample_data.obs_data_types:
                    continue
                data = self._data[data_type] if t is None else self._data[data_type][t,:]
                self.sample_data.pack_data_obs(obs, data, data_types=[data_type])
        return obs
