import numpy as np
from algorithm.cost.cost import Cost

CONSTANT = 1
LINEAR = 2
QUADRATIC = 3
FINAL_ONLY = 4


class CostRamp(Cost):
    """
    A wrapper cost function that applies a ramping multiplier over time

    Args:
        cost: Cost function to ramp
        ramp_option: CONSTANT, LINEAR, QUADRATIC, or FINAL_ONLY. Default CONSTANT
    """

    def __init__(self, cost, ramp_option=CONSTANT):
        Cost.__init__(self, cost._hyperparams, cost.sample_data)
        self.cost = cost
        self.ramp_option = ramp_option

    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError()

    def eval(self, x, u, obs):
        """
        Evaluate time-ramped cost function
        """
        T, _ = x.shape

        l, lx, lu, lxx, luu, lux = self.cost.eval(x, u, obs)

        # Set up time-varying target multiplier.
        if self.ramp_option == CONSTANT:
            wpm = np.ones((T, 1))
        elif self.ramp_option == LINEAR:
            wpm = np.arange(T, dtype=np.float32) / T
        elif self.ramp_option == QUADRATIC:
            wpm = (np.arange(T, dtype=np.float32) / T) ** 2
        elif self.ramp_option == FINAL_ONLY:
            wpm = np.zeros((T, 1))
        else:
            raise ValueError('Unknown cost ramp requested!')

        l *= wpm
        lx *= wpm
        lu *= wpm
        lxx *= wpm
        luu *= wpm
        lux *= wpm
        return l, lx, lu, lxx, luu, lux

