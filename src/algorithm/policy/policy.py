import abc


class Policy(object):
    """
    Computes actions from states/observations
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def act(self, x, obs, t):
        """

        Args:
            x: State vector
            obs: Observation vector
            t: timestep

        Returns:
            A Du-dimensional action vector
        """
        raise NotImplementedError()