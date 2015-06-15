import numpy as np

from algorithm.cost.cost import Cost
from algorithm.cost.cost_utils import evall1l2term


class CostFK(Cost):
    """
    Forward kinematics cost function

    Args:
        hyperparams:
        sample_data:
        wp: 100x9 weight vector
    """

    def __init__(self, hyperparams, sample_data, wp):
        super(CostFK, self).__init__(hyperparams, sample_data)

        # TODO: Discuss how to init parameters
        self.wp = wp
        self.wacc = 1
        self.wprevu = 1
        self.wvel = 1
        self.env_target = True
        self.analytic_jacobian = True

        self.l1 = 0.0
        self.l2 = 1.0
        self.wu = 1e-4
        self.alpha = 1e-2

        self.evalnorm = evall1l2term

    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError()

    # TODO: Currently, sample_meta takes the place of cost_infos. Discuss how to pass these around.
    def eval(self, sample_x, sample_u, sample_obs, sample_meta):
        """
        Evaluate forward kinematics cost.

        Temporary note: This implements the 'joint' penalty type from the matlab code,
            with the velocity/velocity diff/etc. penalties remove (use CostState instead)

        Args:
            sample_X: A T x Dx state matrix
            sample_U: A T x Du action matrix
            sample_Obs: A T x Dobs observation matrix
            sample_meta: List of cost_info objects
                (temporary placeholder until we discuss how to pass these around)
        """
        T, Dx = sample_x.shape
        _, Du = sample_u.shape

        # Initialize terms.
        l = np.zeros((T, 1))
        lu = np.zeros((T, Du))
        lx = np.zeros((T, Dx))
        luu = np.zeros((T, Du, Du))
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        # Choose target.
        if self.env_target:
            tgt = np.reshape(np.concatenate([sample_meta[i].tgt for i in range(len(sample_meta))], axis=1),
                             (sample_meta[0].tgt.shape[0], T))
        else:
            raise NotImplementedError("Must use env_target option")

        # Compute and add FK penalties.
        pt = np.concatenate([sample_meta[i].pt for i in range(len(sample_meta))], axis=1)
        Jx = np.concatenate([sample_meta[i].Jx for i in range(len(sample_meta))], axis=2)
        Jxx = np.concatenate([sample_meta[i].Jxx for i in range(len(sample_meta))], axis=3)

        # Rearrange the points and matrices.
        pt = np.reshape(np.transpose(pt, [0, 2, 1]), (pt.shape[0] * pt.shape[2], pt.shape[1]))
        Jx = np.reshape(np.transpose(Jx, [0, 1, 3, 2]), (Jx.shape[0], Jx.shape[1] * Jx.shape[3], Jx.shape[2]))
        Jxx = np.reshape(np.transpose(Jxx, [0, 1, 2, 4, 3]),
                         (Jxx.shape[0], Jxx.shape[1], Jxx.shape[2] * Jxx.shape[4], Jxx.shape[3]))
        dist = pt - tgt

        # TODO: Remove transposes
        dist = dist.T
        Jx = np.transpose(Jx, [2, 0, 1])
        Jxx = np.transpose(Jxx, [3, 0, 1, 2])

        # Evaluate penalty term.
        if self.analytic_jacobian:
            # Use analytic Jacobians from cost_infos.
            il, ilx, ilxx = self.evalnorm(self.wp, dist, Jx, Jxx, self.l1, self.l2, self.alpha)
        else:
            # Use estimated Jacobians and no higher order terms.
            Jxx_zerod = np.zeros((T, Dx, Dx, dist.shape[0]))
            il, ilx, ilxx = self.evalnorm(self.wp, dist, Jx, Jxx_zerod, self.l1, self.l2, self.alpha)

        # Add to current terms.
        l = l + il
        lx = lx + ilx
        lxx = lxx + ilxx

        return l, lx, lu, lxx, luu, lux


def __doctest():
    """
    Quick doctest just to make sure this runs.

    >>> wp = np.ones((100, 9))
    >>> c = CostFK(None, None, wp)
    >>> X = np.ones((100,39))
    >>> U = np.ones((100,7))
    >>> sample_meta = [lambda x: None for _ in range(100)]
    >>> for i in range(len(sample_meta)):
    ...    sample_meta[i].pt = np.ones((3, 1, 3))*2
    ...    sample_meta[i].tgt = np.ones((9, 1))
    ...    sample_meta[i].Jx = np.ones((39, 3, 1, 3))
    ...    sample_meta[i].Jxx = np.ones((39, 39, 3, 1, 3))
    >>> l, lx, lu, lxx, luu, lux = c.eval(X, U, None, sample_meta)
    >>> np.sum(l)
    450.0
    >>> np.sum(lx)
    35100.0
    >>> np.sum(lu)
    0.0
    >>> np.sum(lxx)
    2737800.0
    >>> np.sum(luu)
    0.0
    >>> np.sum(lux)
    0.0
    """
    pass
