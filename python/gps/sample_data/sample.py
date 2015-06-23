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
        self.dObs = sample_data.dObs

        # Dictionary containing the sample data from various sensors
        self._data = {}

        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._obs = np.empty((self.T, self.dObs))
        self._obs.fill(np.nan)

    def set(self, sensor_name, sensor_data):
        """Set trajectory data for a particular sensor"""
        self._data[sensor_name] = sensor_data
        self._X.fill(np.nan)  # Invalidate existing X
        self._obs.fill(np.nan)  # Invalidate existing Obs

    def get(self, sensor_name):
        """Get trajectory data for a particular sensor"""
        return self._data[sensor_name]

    def get_X(self):
        """Get the state. Put it together if not already precomputed."""
        if np.any(np.isnan(self._X)):
            for data_type in self._data:
                self.sample_data.pack_data_x(self._X, self._data[data_type], data_types=[data_type])
        return self._X

    def get_U(self):
        """Get the action. Put it together if not already precomputed."""
        return self._data[Action]

    def get_obs(self):
        """Get the feature representation. Put it together if not already precomputed."""
        if np.any(np.isnan(self._obs)):
            # TODO: Filter out data not in obs - Maybe this can be handled in pack_data_obs?
            for data_type in self._data:
                self.sample_data.pack_data_obs(self._obs, self._data[data_type], data_types=[data_type])
        return self._obs
