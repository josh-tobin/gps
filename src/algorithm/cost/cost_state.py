import numpy as np

from algorithm.cost.cost import Cost
from algorithm.cost.cost_utils import evall1l2term

class CostState(Cost):
    """
    Computes l1/l2 distance to a fixed target state

    Args:
        hyperparams:
        sample_data:
        tgtstate: 1 x dX Target state vector
        wp: 1 x dX weight vector
    """
    def __init__(self, hyperparams, sample_data, tgtstate, wp):
        super(CostState, self).__init__(hyperparams, sample_data)
        self.tgtstate = tgtstate
        self.wp = wp
        self.l1 = 0.0
        self.l2 = 1.0
        self.alpha = 1e-2

    def update(self):
        """ Update cost values and derivatives. """
        # Compute and add state penalty.
        #TODO: Need a way to select samples to evaluated
        sample_X = self.sample_data.get_X([-1])[0]

        T, Dx = sample_X.shape

        #for sample_idx in range(N):
        dist = sample_X - self.tgtstate

        # Evaluate penalty term.
        l,lx,lxx = evall1l2term(
            np.tile(self.wp, [T, 1]),
            dist,
            np.tile(np.eye(Dx), [T, 1, 1]),
            np.zeros((T,Dx,Dx,Dx)),
            self.l1,
            self.l2,
            self.alpha);
