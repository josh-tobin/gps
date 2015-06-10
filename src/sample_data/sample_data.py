#!/usr/bin/env python

class SampleData():
    """Class that handles writes and reads to sample data.

    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        raise NotImplementedError("TODO");

    def add_samples(self):
        """ Add newly collected samples. """
        raise NotImplementedError("TODO");

    def get_trajs(self):
        """ Return trajectory objects, with phi, X, and U. """
        raise NotImplementedError("TODO");
