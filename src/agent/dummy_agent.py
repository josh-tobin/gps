import numpy as np

from agent import Agent
from sample_data.sample import Sample

class DummyAgent(Agent):
    """
    A dummy agent that generates fake, unrealistic samples.
    Useful for checking if code runs.
    """
    def __init__(self, hyperparams, common_hyperparams, sample_data, state_assembler):
        super(DummyAgent, self).__init__(hyperparams, common_hyperparams, sample_data, state_assembler)
        self.filler = np.random.randn

    def sample(self, N):
        """
        Add samples to sample_data

        Args:
            N: Number of samples to take
        """
        samples = [self.__generate_random_sample() for _ in range(N)]
        self.sample_data.add_samples(samples)

    def __generate_random_sample(self):
        """ Fill X, U, and Phi with random data """
        sample = Sample(self._hyperparams)
        sample._X = self.filler(sample.T, sample.dX)
        sample._U = self.filler(sample.T, sample.dU)
        sample._obs = self.filler(sample.T, sample.dObs)
        return sample

    def test(self):
        raise NotImplementedError();
