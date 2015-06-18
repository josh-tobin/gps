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

    def pack_data_x(self, existing_mat, data_to_insert, data_types=None, axes=None):
        """
        Inserts data into existing_mat into the indices specified by data_types and axes.

        Args:
            existing_mat: Matrix to insert into
            data_to_insert: Matrix to insert into existing_mat.
            data_types (list, string): Name of the sensors you are inserting data for
            axis (list, int): (Optional) Which axis you wish to insert data into.
                Defaults to last axes : -1, -2, ... -len(data_types).

        Example Usage:
        >>> dX = 3; T=2
        >>> sample = Sample({'T':T, 'dX': dX, 'dU': 0, 'dObs': dX})
        >>> sample._sensor_idx = {'a': [0], 'b': [1], 'c': [2]}
        >>> existing_mat = np.zeros((T, dX, dX))
        >>> data_to_insert = np.ones((T, 1, dX))
        >>> sample.pack_data_x(existing_mat, data_to_insert, sensors=['a'], axes=[1])
        >>> existing_mat
        array([[[ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]],
        <BLANKLINE>
               [[ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]])
        >>> data_to_insert = np.ones((T, 1, 1))*2
        >>> sample.pack_data_x(existing_mat, data_to_insert, sensors=['a', 'b'], axes=[1, 2])
        >>> existing_mat
        array([[[ 1.,  2.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]],
        <BLANKLINE>
               [[ 1.,  2.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]])
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume you are indexing on last dimensions
            axes = list(range(-1, -num_sensor-1, -1))
        else:
            # Make sure number of sensors and axes are consistent
            assert num_sensor == len(axes)

        #Shape Checks
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            assert existing_mat.shape[axes[i]] == self.dX  # Make sure you are slicing along X
            insert_shape[axes[i]] = len(self._sensor_idx[data_types[i]])
        # Make sure data is the right shape
        assert tuple(insert_shape) == data_to_insert.shape

        # Actually perform the slice
        index = [slice(None)]*len(existing_mat.shape)
        for i in range(num_sensor):
            # Use slice object to preserve dimensions, rather than just self._sensor_idx
            # Note: this requires that sensor_idx be contiguous and sorted
            index[axes[i]] = slice(self._sensor_idx[data_types[i]][0], self._sensor_idx[data_types[i]][-1]+1)
        existing_mat[index] = data_to_insert

    def unpack_data_x(self, existing_mat, data_types=None, axes=None):
        """
        Returns the data from existing_mat corresponding to data_types.

        Args:
            existing_mat: Matrix to unpack from
            data_types (list, string): Name of the sensor you are unpacking
            axes (list, int): (Optional) Which axes you wish to unpack along.
                Defaults to last axes : -1, -2, ... -len(data_types).
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume you are indexing on last dimensions
            axes = list(range(-1, -num_sensor-1, -1))
        else:
            # Make sure number of sensors and axes are consistent
            assert num_sensor == len(axes)

        #Shape Checks
        for i in range(num_sensor):
            assert existing_mat.shape[axes[i]] == self.dX  # Make sure you are slicing along X

        # Actually perform the slice
        index = [slice(None)]*len(existing_mat.shape)
        for i in range(num_sensor):
            # Use slice object to preserve dimensions, rather than just self._sensor_idx
            # Note: this requires that sensor_idx be contiguous and sorted
            index[axes[i]] = slice(self._sensor_idx[data_types[i]][0], self._sensor_idx[data_types[i]][-1]+1)
        return existing_mat[index]

