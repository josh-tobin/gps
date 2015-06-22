import abc


class Policy(object):
    """
    Computes actions from states/observations
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def act(self, x, obs, t, noise):
        """

        Args:
            x: State vector
            obs: Observation vector
            t: timestep
            noise: A U-dimensional noise vector.

        Returns:
            A Du-dimensional action vector
        """
        raise NotImplementedError()