import numpy as np

from gps.agent.agent import Agent
from gps.sample.sample import Sample


class DummyAgent(Agent):
    """
    A dummy agent that generates fake, unrealistic samples.
    Useful for checking if code runs.
    """
    def __init__(self, hyperparams):
        Agent.__init__(self, hyperparams)
        self.filler = np.random.randn

    def sample(self, policy, condition):
        """
        Generates a random sample.
        """
        sample = Sample(self._hyperparams)
        sample._X = self.filler(T, sample.dX)
        sample._U = self.filler(T, sample.dU)
        sample._obs = self.filler(T, sample.dObs)
        self._samples[condition].append(sample)

    def reset(self, condition):
        pass
