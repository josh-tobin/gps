import numpy as np

from cost import Cost
from cost_utils import evall1l2term, get_ramp_multiplier


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
