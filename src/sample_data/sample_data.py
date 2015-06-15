#!/usr/bin/env python
import cPickle
import numpy as np

class SampleData(object):
    """Class that handles writes and reads to sample data.

    """
    def __init__(self, hyperparams, common_hyperparams, state_assembler, sample_writer=None):
        self._hyperparams = hyperparams
        #self._experiment_dir = common_hyperparams['experiment_dir']
        #self._data_file = self._experiment_dir + hyperparams['filename']
        self.sample_writer = sample_writer

        # List of trajectory samples (roll-outs)
        self._samples = []

    def get_X(self, idx):
        """Returns NxTxdX numpy array of states"""
        # TODO - check above dimensions
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx):
        """Returns NxTxdU numpy array of actions"""
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_phi(self, idx):
        """Returns NxTxdPhi numpy array of feature representations"""
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def add_samples(self, samples):
        """ Add newly collected samples. Save out new samples."""

        if not isinstance(samples,list):
            samples = [samples]

        self._samples.extend(samples)

        if self.sample_writer:
            self.sample_writer.write(self._samples)

class PickleSampleWriter(object):
    """ Pickles samples into data_file """
    def __init__(self, data_file):
        self._data_file = data_file

    def write(self, samples):
        with open(self._data_file, 'wb') as data_file:
            cPickle.dump(data_file, self._samples)
