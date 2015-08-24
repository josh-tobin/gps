import numpy as np

from algorithm.cost.cost_utils import get_ramp_multiplier, RAMP_QUADRATIC, RAMP_LINEAR, RAMP_CONSTANT, evall1l2term, evall1l2term_fast, evallogl2term_fast

class CostStateTracking(object):
    def __init__(self, wp, tgt, maxT=None):
        self.wp = wp
        self.mu = tgt
        self.ref_len = tgt.shape[0]
        self.ramp_option = RAMP_CONSTANT
        self.final_tgt = True
        self.final_penalty = 10.0  # weight = sum of remaining weight * final penalty
        self.t_weight = 10.0
        self.l1 = 0.01
        self.l2 = 1.0
        self.alpha = 1e-5
        self.wu = 10e-3/np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])  # Brett CostFK Big least squares

        #self.wu = 1e-2/np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])  # MJC

        ramp_len = self.ref_len if maxT is None else maxT
        self.wpm = get_ramp_multiplier(self.ramp_option, ramp_len, wp_final_multiplier=1.0)

    def eval(self, X, U, t, jac=None):
        # Constants.
        dX = X.shape[1]
        dU = U.shape[1]
        T = X.shape[0]

        wp = self.wp[:dX]*np.expand_dims(self.wpm[t:t+T], axis=-1)
        remaining_weight = np.sum(self.wpm[t+T:])
        wp[-1,:] *= self.final_penalty

        #l = np.zeros(T)
        l = 0.5 * np.sum(self.wu * (U ** 2), axis=1)
        lu = self.wu*U
        #lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self.wu), [T, 1, 1])
        #lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, dU, dX))

        # Compute target based on nearest neighbors
        tgt = np.zeros((T, dX))
        Tmu = self.mu.shape[0]
        cand_idx = np.zeros(T)
        t_ramp = self.t_weight*np.arange(t,t+T)
        query_pnts = np.c_[X*self.wp, t_ramp]
        tgt_points = np.c_[self.mu[:,:dX]*self.wp, self.t_weight*np.arange(Tmu)]
        for i in range(T):
            min_idx = nearest_neighbor(query_pnts[i], tgt_points)
            if self.final_tgt:
            	min_idx = self.ref_len-1
            cand_idx[i] = min_idx
            tgt[i] = self.mu[min_idx,:dX]
        dist = X - tgt
        # Evaluate penalty term.
        #l, lx, lxx = evall1l2term( wp, dist, np.tile(np.eye(dX), [T, 1, 1]), np.zeros((T, dX, dX, dX)),
        #    self.l1, self.l2, self.alpha)

        if wp.shape != tgt.shape:
            import pdb; pdb.set_trace()
        l, lx, lxx = evallogl2term_fast( wp, dist, self.l1, self.l2, self.alpha)

        return l, lx, lu, lxx, luu, lux

    def compute_nearest_neighbors(self, X, U, t):
    	T = X.shape[0]
        Tmu = self.mu.shape[0]
        cand_idx = np.zeros(T)
        t_ramp = self.t_weight*np.arange(t,t+T)
        query_pnts = np.c_[X, t_ramp]
        tgt_points = np.c_[self.mu, self.t_weight*np.arange(Tmu)]
        for i in range(T):
            min_idx = nearest_neighbor(query_pnts[i], tgt_points)
            cand_idx[i] = min_idx
        return cand_idx

    def get_ee_tgt(self, t):
    	#return self.mu[t,21:30]
        return self.mu[t,21-7:30-7]

def nearest_neighbor(query_pnt, tgt_points):
    dist = tgt_points - query_pnt
    dist = np.sum( dist*dist, axis=1)
    min_idx = np.argmin(dist)
    return min_idx
