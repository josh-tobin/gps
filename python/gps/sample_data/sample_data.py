import cPickle
import os
import numpy as np

from gps_sample_types import *
from sample import Sample


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
        self.dO = self._hyperparams['dO']

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']
        #TODO: not sure how we were planning on determining correct indices for x and obs
        #      leaving it as a hyperparameter for now, just for testing purposes
        # List of indices for each data type in state X. Indices must be contiguous
        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,  self._hyperparams['state_idx'])}
        # List of indices for each data type in observation. Indices must be contiguous
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types,  self._hyperparams['obs_idx'])}

        # List of trajectory samples (roll-outs)
        self._samples = []

    def create_new(self):
        """Construct and return a new sample and add it to the list of samples."""
        sample = Sample(self)
        self._samples.append(sample)
        return sample

    def get_X(self, idx=None):
        """Returns NxTxdX numpy array of states"""
        # TODO - check above dimensions
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx=None):
        """Returns NxTxdU numpy array of actions"""
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_obs(self, idx=None):
        """Returns NxTxdO numpy array of feature representations"""
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def get_samples(self, idx=None):
        """ Returns N sample objects """
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def num_samples(self):
        return len(self._samples)

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
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume you are indexing on last dimensions
            axes = list(range(-1, -num_sensor-1, -1))
        else:
            # Make sure number of sensors and axes are consistent
            if num_sensor != len(axes):
                raise ValueError('Length of sensors (%d) must equal length of axes (%d)', num_sensor, len(axes))

        #Shape Checks
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            if existing_mat.shape[axes[i]] != self.dO:  # Make sure you are slicing along X
                raise ValueError('Axes must be along an dX=%d dimensional axis', self.dO)
            insert_shape[axes[i]] = len(self._obs_data_idx[data_types[i]])
        # Make sure data is the right shape
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s', data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice
        index = [slice(None)]*len(existing_mat.shape)
        for i in range(num_sensor):
            index[axes[i]] = slice(self._obs_data_idx[data_types[i]][0], self._obs_data_idx[data_types[i]][-1]+1)
        existing_mat[index] = data_to_insert

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
        >>> sample_data = SampleData({'T':T, 'dX': dX, 'dU': 0, 'dO': dX}, None, SysOutWriter())
        >>> sample_data._x_data_idx = {'a': [0], 'b': [1], 'c': [2]}
        >>> existing_mat = np.zeros((T, dX, dX))
        >>> data_to_insert = np.ones((T, 1, dX))
        >>> sample_data.pack_data_x(existing_mat, data_to_insert, data_types=['a'], axes=[1])
        >>> existing_mat
        array([[[ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]],
        <BLANKLINE>
               [[ 1.,  1.,  1.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]]])
        >>> data_to_insert = np.ones((T, 1, 1))*2
        >>> sample_data.pack_data_x(existing_mat, data_to_insert, data_types=['a', 'b'], axes=[1, 2])
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
            if num_sensor != len(axes):
                raise ValueError('Length of sensors (%d) must equal length of axes (%d)' % (num_sensor, len(axes)))

        #Shape Checks
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            if existing_mat.shape[axes[i]] != self.dX:  # Make sure you are slicing along X
                raise ValueError('Axes must be along an dX=%d dimensional axis', self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        # Make sure data is the right shape
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s' % (data_to_insert.shape, tuple(insert_shape)))

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
            if num_sensor != len(axes):
                raise ValueError('Length of sensors (%d) must equal length of axes (%d)', num_sensor, len(axes))

        #Shape Checks
        for i in range(num_sensor):
            if existing_mat.shape[axes[i]] != self.dX:  # Make sure you are slicing along X
                raise ValueError('Axes must be along an dX=%d dimensional axis', self.dX)

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
