#!/usr/bin/env python

class Agent(object):
    """Agent superclass

    """
    def __init__(self, hyperparams, common_hyperparams, sample_data, state_assembler):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.state_assembler = state_assembler

    def sample(self, N):
        raise NotImplementedError("Must be implemented in subclass");

    def test(self):
        raise NotImplementedError("Must be implemented in subclass");
