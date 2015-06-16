import numpy as np

from cost import Cost
from cost_utils import evallogl2term, get_ramp_multiplier, RAMP_CONSTANT


class CostFK(Cost):
    """
    Forward kinematics cost function

    Args:
        hyperparams:
        sample_data:
        wp: 1x9 weight vector for each end-effector point
    """

    def __init__(self, hyperparams, sample_data, wp):
        super(CostFK, self).__init__(hyperparams, sample_data)

        # TODO: Discuss how to init parameters
        self.wp = wp
        self.env_target = True
        self.analytic_jacobian = True
        self.ramp_option = RAMP_CONSTANT

        self.l1 = 0.1
        self.l2 = 0.0001
        self.alpha = 1e-5

        self.evalnorm = evallogl2term
        self.wp_final_multiplier = 100

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

        wpm = get_ramp_multiplier(self.ramp_option, T, wp_final_multiplier=self.wp_final_multiplier)
        wp = self.wp*np.expand_dims(wpm, axis=-1)

        # Initialize terms.
        l = np.zeros((T, 1))
        lu = np.zeros((T, Du))
        lx = np.zeros((T, Dx))
        luu = np.zeros((T, Du, Du))
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))

        # Choose target.
        if self.env_target:
            dim1, _, dim2 = sample_meta[0].tgt.shape
            tgt = np.concatenate([sample_meta[i].tgt for i in range(len(sample_meta))], axis=1)
            tgt = np.transpose(tgt, [1, 0, 2])
            tgt = np.reshape(tgt, (T, dim1*dim2))
            tgt = tgt.T  # TODO: remove extra tranposes
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
            il, ilx, ilxx = self.evalnorm(wp, dist, Jx, Jxx, self.l1, self.l2, self.alpha)
        else:
            # Use estimated Jacobians and no higher order terms.
            Jxx_zerod = np.zeros((T, Dx, Dx, dist.shape[0]))
            il, ilx, ilxx = self.evalnorm(wp, dist, Jx, Jxx_zerod, self.l1, self.l2, self.alpha)

        # Add to current terms.
        l = l + il
        lx = lx + ilx
        lxx = lxx + ilxx

        return l, lx, lu, lxx, luu, lux
