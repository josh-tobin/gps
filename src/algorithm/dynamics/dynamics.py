#!/usr/bin/env python

class Dynamics():
    """Dynamics superclass

    """
    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self._sample_data = sample_data

    def update(self):
        """ Update dynamics. """
        raise NotImplementedError("Must be implemented in subclass");

    def eval(self):
        """ Evaluate dynamics. """
        raise NotImplementedError("Must be implemented in subclass");
