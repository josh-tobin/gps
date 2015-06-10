#!/usr/bin/env python

class TrajOpt():
    """Trajectory optimization superclass

    """
    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    def update(self):
        """ Update trajectory distributions. """
        raise NotImplementedError("Must be implemented in subclass");

# TODO - need interface with C++ trajopt
