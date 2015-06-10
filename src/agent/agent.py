#!/usr/bin/env python

class Agent():
    """Agent superclass

    """
    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    def sample(self):
        raise NotImplementedError("Must be implemented in subclass");

    def test(self):
        raise NotImplementedError("Must be implemented in subclass");
