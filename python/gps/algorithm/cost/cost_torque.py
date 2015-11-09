from copy import deepcopy
import numpy as np

from gps.algorithm.cost.config import cost_torque
from gps.algorithm.cost.cost import Cost


class CostTorque(Cost):
    """
    Computes torque penalties
    """

    def __init__(self, hyperparams):
        config = deepcopy(cost_torque)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample

        Args:
            sample: A single sample
        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        lu = self._hyperparams['wu']*sample_u
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux
