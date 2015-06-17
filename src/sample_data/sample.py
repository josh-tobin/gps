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
        self.dObs = self._hyperparams['dObs']

        # list of numpy arrays containing the sample data from various sensors
        self._data = []

        # To be populated in by the C++ object, maps sensor name to index
        self._sensor_idx = {}  # Actually now just DataType object

        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._U = np.empty((self.T, self.dU))
        self._U.fill(np.nan)
        self._obs = np.empty((self.T, self.dObs))
        self._obs.fill(np.nan)

    def set(self, sensor_name, sensor_data):
        """Set trajectory data for a particular sensor"""
        self._data[self._sensor_idx[sensor_name]] = sensor_data

    def get(self, sensor_name):
        """Get trajectory data for a particular sensor"""
        return self._data[self._sensor_idx[sensor_name]]

    def get_X(self):
        """Get the state. Put it together if not already precomputed."""
        if np.any(np.isnan(self._X)):
            raise NotImplementedError("TODO - Compute _X by calling C++ code")
        return self._X

    def get_U(self):
        """Get the action. Put it together if not already precomputed."""
        if np.any(np.isnan(self._U)):
            raise NotImplementedError("TODO - Compute _U by calling C++ code")
        return self._U

    def get_obs(self):
        """Get the feature representation. Put it together if not already precomputed."""
        if np.any(np.isnan(self._obs)):
            raise NotImplementedError("TODO - Compute _obs by calling C++ code")
        return self._obs

    def pack_data_x(self, existing_mat, sensor_name, data_to_insert, axis=-1):
        """
        Inserts data into existing_mat corresponding to sensor_name along a state (X) dimension.

        Args:
            existing_mat: Matrix to insert into
            sensor_name: Name of the sensor you are inserting data for
            axis: Which axis you wish to insert data into
        """
        #Shape Checks
        assert existing_mat.shape[axis] == self.dX  # Make sure you are slicing along X
        insert_shape = list(existing_mat.shape)
        insert_shape[axis] = len(self._sensor_idx[sensor_name])
        assert tuple(insert_shape) == data_to_insert.shape  # Make sure data is the right shape

        index = [slice(None)]*len(existing_mat.shape)
        index[axis] = self._sensor_idx[sensor_name]
        existing_mat[index] = data_to_insert

    def slice_data_x(self, existing_mat, sensor_name, axis=-1):
        """
        Returns the data from existing_mat corresponding to sensor_name along a state (X) dimension.

        Args:
            existing_mat: Matrix to slice data out of
            sensor_name: Name of values to pull out of existing_mat
            axis: Which axis you wish to pull data out of

        Returns:
            A slice matrix. All dimensions will be the same except for the sliced
            dimension, which is equal to how many dimensions the sensor has.
        """
        #Shape Checks
        assert existing_mat.shape[axis] == self.dX  # Make sure you are slicing along X

        index = [slice(None)]*len(existing_mat.shape)
        index[axis] = self._sensor_idx[sensor_name]
        return existing_mat[index]

