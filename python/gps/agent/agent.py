import abc
from gps.sample.sample_list import SampleList

from gps.proto.gps_pb2 import ACTION


class Agent(object):
    """Agent superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

        # Store samples, along with size and index information for samples
        self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        self.T = self._hyperparams['T']
        self.dU = self._hyperparams['sensor_dims'][ACTION]

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']
        # list of indices for each data type in state X
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX = i
        # list of indices for each data type in observation
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO = i
        self._x_data_idx = {d: i for d, i in zip(self.x_data_types, self._state_idx)}
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types, self._obs_idx)}

    @abc.abstractmethod
    def sample(self, policy, condition):
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def reset(self, condition):
        """
        Reset the agent to be ready for a particular experiment condition.

        Args:
            condition (int): Integer code for which experiment condition to set up.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_samples(self, condition, start=0, end=None):
        """
        Return the requested samples based on the start and end indices.

        Args:
            start (int): Starting index of samples to return.
            end (int): End index of samples to return.
        """
        return SampleList(self._samples[condition][start:]) if end == None \
                else SampleList(self._samples[condition][start:end])

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
        #TODO: update 'Example Usage' below, now that dX, etc. aren't passed in
        """
        Inserts data into existing_mat into the indices specified by data_types and axes.
        Can insert 1 data type per axis.

        Args:
            existing_mat: Matrix to insert into
            data_to_insert: Matrix to insert into existing_mat.
            data_types (list, string): Name of the sensors you are inserting data for
            axis (list, int): (Optional) Which axis you wish to insert data into.
                Defaults to last axes : -1, -2, ... -len(data_types).

        TODO: Update/remove the following example.
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
