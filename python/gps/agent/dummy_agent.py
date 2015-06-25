import numpy as np

from agent import Agent
from sample_data.sample import Sample


class DummyAgent(Agent):
    """
    A dummy agent that generates fake, unrealistic samples.
    Useful for checking if code runs.
    """
    def __init__(self, hyperparams, sample_data, state_assembler):
        super(DummyAgent, self).__init__(hyperparams, sample_data, state_assembler)
        self.filler = np.random.randn

    def sample(self, policy, T):
        """
        Returns a random sample.
        """
        sample = Sample(self._hyperparams)
        sample._X = self.filler(T, sample.dX)
        sample._U = self.filler(T, sample.dU)
        sample._obs = self.filler(T, sample.dObs)
        return sample

    def reset(self, condition):
        pass
