from copy import deepcopy
import numpy as np

from config import cost_torque
from cost import Cost


class CostTorque(Cost):
    """
    Computes torque penalties
    """

    def __init__(self, hyperparams, sample_data):
        config = deepcopy(cost_torque)
        config.update(hyperparams)
        Cost.__init__(self, config, sample_data)

    def eval(self, sample_idx):
        """
        Evaluate cost function and derivatives on a sample

        Args:
            sample_idx: A single index into sample_data
        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        sample_u = self.sample_data.get_U([sample_idx])[0,:,:]
        T = self.sample_data.T
        Du = self.sample_data.dU
        Dx = self.sample_data.dX
        l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)
        lu = self._hyperparams['wu']*sample_u
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux
