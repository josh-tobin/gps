import abc


class Agent(object):
    """Agent superclass

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, sample_data):
        self._hyperparams = hyperparams
        self.sample_data = sample_data

    @abc.abstractmethod
    def sample(self, policy, T):
        raise NotImplementedError("Must be implemented in subclass")

    @abc.abstractmethod
    def reset(self, condition):
        """
        Reset the agent to be ready for a particular experiment condition.

        Args:
            condition (int): Integer code for which experiment condition to set up.
        """
        raise NotImplementedError("Must be implemented in subclass")
