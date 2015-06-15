import abc

class Agent(object):
    """Agent superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, common_hyperparams, sample_data, state_assembler):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.state_assembler = state_assembler

    @abc.abstractmethod
    def sample(self, N):
        raise NotImplementedError("Must be implemented in subclass");

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError("Must be implemented in subclass");
