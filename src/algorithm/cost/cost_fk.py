import numpy as np

from cost import Cost
from cost_utils import get_ramp_multiplier


class CostFK(Cost):
    """
    Forward kinematics cost function. Used for costs involving the
    end effector position.
    """

    def __init__(self, hyperparams, sample_data):
        config.update(hyperparams)

        Cost.__init__(self, config, sample_data)

    def eval(self, sample):
        """
        Evaluate forward kinematics (end-effector penalties) cost.

        Temporary note: This implements the 'joint' penalty type from the matlab code,
            with the velocity/velocity diff/etc. penalties remove (use CostState instead)

        Args:
            sample: A Sample object

        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        T = sample.T
        dX = sample.dX
        dU = sample.dU

        wpm = get_ramp_multiplier(self._hyperparams['ramp_option'], T,
                                  wp_final_multiplier=self._hyperparams['wp_final_multiplier'])
        wp = self._hyperparams['wp'] * np.expand_dims(wpm, axis=-1)

        # Initialize terms.
        l = np.zeros((T,))
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        # Choose target.
        if self._hyperparams['env_target']:
            tgt = sample.get('EndEffectorTarget')
        else:
            raise NotImplementedError('Must use env_target option')

        pt = sample.get('EndEffectorPoint')
        dist = pt - tgt
        jx = sample.get('EndEffectorJacobian')

        # Evaluate penalty term.
        if self._hyperparams['analytic_jacobian']:
            jxx = sample.get('EndEffector2ndJacobian')
            il, ilx, ilxx = self._hyperparams['evalnorm'](wp, dist, jx, jxx,
                                                          self._hyperparams['l1'],
                                                          self._hyperparams['l2'],
                                                          self._hyperparams['alpha'])
        else:
            # Use estimated Jacobians and no higher order terms
            jxx_zeros = np.zeros((T, dist.shape[0], dX, dX))
            il, ilx, ilxx = self._hyperparams['evalnorm'](wp, dist, jx, jxx_zeros,
                                                          self._hyperparams['l1'],
                                                          self._hyperparams['l2'],
                                                          self._hyperparams['alpha'])
        # Add to current terms.
        l = l + il
        lx = lx + ilx
        lxx = lxx + ilxx

        return l, lx, lu, lxx, luu, lux
