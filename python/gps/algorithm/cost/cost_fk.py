from copy import deepcopy
import numpy as np

from config import cost_fk
from cost import Cost
from cost_utils import get_ramp_multiplier
from proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_HESSIANS, END_EFFECTOR_JACOBIANS


class CostFK(Cost):
    """
    Forward kinematics cost function. Used for costs involving the
    end effector position.
    """

    def __init__(self, hyperparams, sample_data):
        config = deepcopy(cost_fk)
        config.update(hyperparams)
        Cost.__init__(self, config, sample_data)

    def eval(self, sample_idx):
        """
        Evaluate forward kinematics (end-effector penalties) cost.

        Temporary note: This implements the 'joint' penalty type from the matlab code,
            with the velocity/velocity diff/etc. penalties remove (use CostState instead)

        Args:
            sample_idx: A single index into sample_data

        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        T = self.sample_data.T
        dX = self.sample_data.dX
        dU = self.sample_data.dU
        sample = self.sample_data.get_samples(idx=[sample_idx])[0]

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
        tgt = self._hyperparams['end_effector_target']
        pt = sample.get(END_EFFECTOR_POINTS)
        dist = pt - tgt
        jx = sample.get(END_EFFECTOR_JACOBIANS)

        # Evaluate penalty term.
        if self._hyperparams['analytic_jacobian']:
            jxx = sample.get(END_EFFECTOR_HESSIANS)
            il, ilx, ilxx = self._hyperparams['evalnorm'](wp, dist, jx, jxx,
                                                          self._hyperparams['l1'],
                                                          self._hyperparams['l2'],
                                                          self._hyperparams['alpha'])
        else:
            # Use estimated Jacobians and no higher order terms
            jxx_zeros = np.zeros((T, dist.shape[1], dX, dX))
            il, ilx, ilxx = self._hyperparams['evalnorm'](wp, dist, jx, jxx_zeros,
                                                          self._hyperparams['l1'],
                                                          self._hyperparams['l2'],
                                                          self._hyperparams['alpha'])
        # Add to current terms.
        l = l + il
        lx = lx + ilx
        lxx = lxx + ilxx

        return l, lx, lu, lxx, luu, lux
