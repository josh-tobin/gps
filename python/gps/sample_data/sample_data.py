import cPickle
import os
import numpy as np
from gps_sample_types import *


class SampleData(object):
    """Class that handles writes and reads to sample data.

    """

    def __init__(self, hyperparams, common_hyperparams, sample_writer=None):
        self._hyperparams = hyperparams

        if sample_writer is None:
            experiment_dir = common_hyperparams['experiment_dir']
            data_file = os.path.join(experiment_dir, hyperparams['filename'])
            self.sample_writer = PickleSampleWriter(data_file)
        else:
            self.sample_writer = sample_writer

        self.T = self._hyperparams['T']
        self.dX = self._hyperparams['dX']
        self.dU = self._hyperparams['dU']
        self.dObs = self._hyperparams['dObs']

        self._x_data_idx = {}  # List of indices for each data type in state X. Indices must be contiguous
        self._obs_data_idx = {}  # List of indices for each data type in observation. Indices must be contiguous

        # List of trajectory samples (roll-outs)
        self._samples = []

    def get_X(self, idx):
        """Returns NxTxdX numpy array of states"""
        # TODO - check above dimensions
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx):
        """Returns NxTxdU numpy array of actions"""
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_obs(self, idx):
        """Returns NxTxdObs numpy array of feature representations"""
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def add_samples(self, samples):
        """ Add newly collected samples. Save out new samples."""

        if not isinstance(samples, list):
            samples = [samples]

        self._samples.extend(samples)

        if self.sample_writer:
            self.sample_writer.write(self._samples)

    def get_idx(self, data_type):
        """
        Returns the indices in the state occupied by data_type
        Args:
            data_type (string): Name of sensor
        """
        return self._data_idx[data_type]

    def pack_data_obs(self, existing_mat, data_to_insert, data_types=None, axes=None):
        raise NotImplementedError()

    def pack_data_x(self, existing_mat, data_to_insert, data_types=None, axes=None):
        """
        Inserts data into existing_mat into the indices specified by data_types and axes.
        Can insert 1 data type per axis.

        Args:
            existing_mat: Matrix to insert into
            data_to_insert: Matrix to insert into existing_mat.
            data_types (list, string): Name of the sensors you are inserting data for
            axis (list, int): (Optional) Which axis you wish to insert data into.
                Defaults to last axes : -1, -2, ... -len(data_types).

        Example Usage:
        >>> dX = 3; T=2
        >>> sample_data = SampleData({'T':T, 'dX': dX, 'dU': 0, 'dObs': dX}, None, SysOutWriter())
        >>> sample_data._data_idx = {'a': [0], 'b': [1], 'c': [2]}
        >>> existing_mat = np.zeros((T, dX, dX))
        >>> data_to_insert = np.ones((T, 1, dX))
        >>> sample_data.pack_data_x(existing_mat, data_to_insert, sensors=['a'], axes=[1])
        >>> existing_mat
        array([[[ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]],
        <BLANKLINE>
               [[ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]])
        >>> data_to_insert = np.ones((T, 1, 1))*2
        >>> sample_data.pack_data_x(existing_mat, data_to_insert, sensors=['a', 'b'], axes=[1, 2])
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
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        # Make sure data is the right shape
        assert tuple(insert_shape) == data_to_insert.shape

        # Actually perform the slice
        index = [slice(None)]*len(existing_mat.shape)
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0], self._x_data_idx[data_types[i]][-1]+1)
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
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0], self._x_data_idx[data_types[i]][-1]+1)
        return existing_mat[index]


class PickleSampleWriter(object):
    """ Pickles samples into data_file """

    def __init__(self, data_file):
        self._data_file = data_file

    def write(self, samples):
        with open(self._data_file, 'wb') as data_file:
            cPickle.dump(data_file, samples)


class SysOutWriter(object):
    """ Writes notifications to system.out on sample writes """

    def __init__(self):
        pass

    def write(self, samples):
        print 'Collected %d samples' % len(samples)
