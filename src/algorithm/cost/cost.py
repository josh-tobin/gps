#!/usr/bin/env python

class Cost():
    """Cost superclass

    """
    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError("Must be implemented in subclass");
