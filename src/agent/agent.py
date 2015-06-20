import abc


class Agent(object):
    """Agent superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data, state_assembler):
        self._hyperparams = hyperparams
        self.sample_data = sample_data
        self.state_assembler = state_assembler

    @abc.abstractmethod
    def sample(self, policy, T):
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def reset(self, condition):
        """
        Reset the agent to be ready for a particular experiment condition.
        """
        raise NotImplementedError("Must be implemented in subclass")
