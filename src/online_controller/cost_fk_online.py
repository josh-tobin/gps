import numpy as np

from algorithm.cost.cost_utils import get_ramp_multiplier, RAMP_CONSTANT, evall1l2term, evall1l2term_fast

class CostStateTracking(object):
    def __init__(self, wp, tgtee):
        self.wp = wp
        self.tgtee = tgtee
        self.ramp_option = RAMP_CONSTANT
        self.t_weight = 0.05
        self.l1 = 0.1
        self.l2 = 10.0
        self.alpha = 1e-5
        #self.wu = 1e-2/np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])
        self.wu = 3e-3/np.array([2.09,1.08,0.393,0.674,0.111,0.152,0.098])

    def eval(self, X, U, t):
        raise NotImplementedError()
