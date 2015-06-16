import numpy as np

from algorithm.cost.cost import Cost
from algorithm.cost.cost_utils import evall1l2term


class CostState(Cost):
    """
    Computes l1/l2 distance to a fixed target state

    Args:
        hyperparams:
        sample_data:
        desired_state: 1 x dX target state vector. Penalties will
            be applied based on difference between measured state and this state.
        wp: 1 x dX weight vector along each dimension. To ignore a dimension,
            set its weight to 0.
    """

    def __init__(self, hyperparams, sample_data, desired_state, wp):
        super(CostState, self).__init__(hyperparams, sample_data)
        self.desired_state = desired_state
        self.wp = wp

        # TODO: Hold off on storing defaults
        self.l1 = 0.0
        self.l2 = 1.0
        self.wu = 1e-4
        self.alpha = 1e-2

    def update(self):
        """ Update cost values and derivatives. """
        # Compute and add state penalty.
        # TODO: Need a way to select samples to evaluated
        sample_X = self.sample_data.get_X([-1])[0]  # Gets the latest sample from sample_data
        sample_U = self.sample_data.get_U([-1])[0]

        self.eval(sample_X, sample_U, None)
        #TODO: Where to store/return loss+derivatives after they are evaluated?

    def eval(self, sample_x, sample_u, sample_obs, sample_meta):
        T, Dx = sample_x.shape
        _, Du = sample_u.shape

        # Compute torque penalty and initialize terms.
        l = 0.5 * self.wu * np.sum(sample_u ** 2, axis=1, keepdims=True)
        lu = self.wu * sample_u
        lx = np.zeros((T, Dx))
        luu = self.wu * np.tile(np.eye(Du), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        # Compute state penalty
        dist = sample_x - self.desired_state

        # Evaluate penalty term.
        il, ilx, ilxx = evall1l2term(
            np.tile(self.wp, [T, 1]),
            dist,
            np.tile(np.eye(Dx), [T, 1, 1]),
            np.zeros((T, Dx, Dx, Dx)),
            self.l1,
            self.l2,
            self.alpha)

        l += il
        lx += ilx
        lxx += ilxx

        return l, lx, lu, lxx, luu, lux

