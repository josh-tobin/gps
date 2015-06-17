import numpy as np

from cost import Cost
from cost_utils import evall1l2term, get_ramp_multiplier


class CostState(Cost):
    """
    Computes l1/l2 distance to a fixed target state
    """

    def __init__(self, hyperparams, sample_data):
        super(CostState, self).__init__(hyperparams, sample_data)
        self.desired_state = hyperparams['desired_state']
        self.wp = hyperparams['wp']
        self.ramp_option = hyperparams['ramp_option']

        self.l1 = hyperparams['l1']
        self.l2 = hyperparams['l2']
        self.wu = hyperparams['wu']
        self.alpha = hyperparams['alpha']
        self.wp_final_multiplier = hyperparams['wp_final_multiplier']

    def eval(self, sample_x, sample_u, sample_obs, sample_meta):
        """
        Evaluate cost function and derivatives

        Args:
            x: A T x Dx state matrix
            u: A T x Du action matrix
            obs: A T x Dobs observation matrix
            sample_meta: List of cost_info objects
                (temporary placeholder until we discuss how to pass these around)
        Return:
            l, lx, lu, lxx, luu, lux: Loss (Tx1 float) and 1st/2nd derivatives.
        """
        T, Dx = sample_x.shape
        _, Du = sample_u.shape

        wpm = get_ramp_multiplier(self.ramp_option, T, wp_final_multiplier=self.wp_final_multiplier)
        wp = self.wp*np.expand_dims(wpm, axis=-1)

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
            wp,
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
