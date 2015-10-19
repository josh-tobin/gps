import cPickle
import os
import numpy as np


class SampleList(object):
    """Class that handles writes and reads to sample data.

    """

    def __init__(self, samples):
        #TODO: figure out writing samples to file, where does it go?
        self._samples = samples

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

    # Convenience methods
    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        return self.get_samples([idx])[0]


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
