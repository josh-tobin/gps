import numpy as np

from cost import Cost


class CostTorque(Cost):
    """
    Computes torque penalties
    """

    def __init__(self, hyperparams, sample_data):
        super(CostTorque, self).__init__(hyperparams, sample_data)
        self.wu = hyperparams['wu']

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample

        Args:
            sample: A Sample object
        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (Tx1 float) and derivatives with respect to states (x) and/or actions (u).
        """
        sample_u = sample.get_U()
        T = sample.get_T()
        Du = sample.get_Du()
        Dx = sample.get_Dx()

        l = 0.5 * self.wu * np.sum(sample_u ** 2, axis=1)
        lu = self.wu * sample_u
        lx = np.zeros((T, Dx))
        luu = self.wu * np.tile(np.eye(Du), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        return l, lu, lx, luu, lxx, lux
