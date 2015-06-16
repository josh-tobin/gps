import numpy as np

from cost import Cost
from cost_utils import evall1l2term
from config import cost_state as config


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

    def __init__(self, hyperparams, sample_data):
        Cost.__init__(self, hyperparams, sample_data)

        config.update(hyperparams)
        self.desired_state = config['desired_state']
        self.wp = config['wp']
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.wu = config['wu']
        self.alpha = config['alpha']

    def eval(self, sample_x, sample_u, sample_obs, sample_meta):
        T, dX = sample_x.shape
        _, dU = sample_u.shape

        # Compute torque penalty and initialize terms.
        l = 0.5 * self.wu * np.sum(sample_u ** 2, axis=1, keepdims=True)
        lu = self.wu * sample_u
        lx = np.zeros((T, dX))
        luu = self.wu * np.tile(np.eye(dU), [T, 1, 1])
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        # Compute state penalty
        dist = sample_x - self.desired_state

        # Evaluate penalty term.
        il, ilx, ilxx = evall1l2term(
            np.tile(self.wp, [T, 1]),
            dist,
            np.tile(np.eye(dX), [T, 1, 1]),
            np.zeros((T, dX, dX, dX)),
            self.l1,
            self.l2,
            self.alpha)

        l += il
        lx += ilx
        lxx += ilxx

        return l, lx, lu, lxx, luu, lux


def __finite_differences_doctest():
    """
    >>> import numpy as np
    >>> import algorithm.cost.cost_utils as cost_utils
    >>> T=30; Dx = 7; Du = 3; Dobs=2
    >>> wp = np.ones((1, Dx))
    >>> desired_state = np.ones((1, Dx))
    >>> X = np.random.randn(T, Dx)
    >>> U = np.random.randn(T, Du)
    >>> Obs = np.random.randn(T, Dobs)
    >>> metadata = [None]*T  # CostState uses no metadata
    >>> cost = CostState(None, None, desired_state, wp)
    >>> cost_utils.finite_differences_cost_test(cost, X, U, Obs, metadata)
    True
    """
    pass
