import numpy as np

from config import cost_state
from cost import Cost
from cost_utils import evall1l2term, get_ramp_multiplier


class CostState(Cost):
    """
    Computes l1/l2 distance to a fixed target state
    """

    def __init__(self, hyperparams):
        config = cost_state.copy()
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate cost function and derivatives on a sample

        Args:
            sample: A Sample object
        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        for data_type_name in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type_name]
            wp = config['wp']
            tgt = config['desired_state']
            x = sample.get(data_type_name)
            _, dim_sensor = x.shape

            wpm = get_ramp_multiplier(self._hyperparams['ramp_option'], T,
                                      wp_final_multiplier=self._hyperparams['wp_final_multiplier'])
            wp = wp*np.expand_dims(wpm, axis=-1)
            # Compute state penalty
            dist = x - tgt

            # Evaluate penalty term.
            l, ls, lss = evall1l2term(
                wp,
                dist,
                np.tile(np.eye(dim_sensor), [T, 1, 1]),
                np.zeros((T, dim_sensor, dim_sensor, dim_sensor)),
                self._hyperparams['l1'],
                self._hyperparams['l2'],
                self._hyperparams['alpha'])

            final_l += l
            sample.pack_data_x(final_lx, ls, data_types=[data_type_name])
            sample.pack_data_x(final_lxx, lss, data_types=[data_type_name, data_type_name])

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
