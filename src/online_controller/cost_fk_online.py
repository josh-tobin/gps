import numpy as np

from algorithm.cost.cost_utils import get_ramp_multiplier, RAMP_CONSTANT, evall1l2term, evallogl2term

class CostFKOnline(object):
    def __init__(self, eetgt, ee_idx=None, jnt_idx=None, maxT=None):
        self.dim_ee = ee_idx.stop-ee_idx.start
        self.dim_jnt = jnt_idx.stop - jnt_idx.start
        self.wp = np.ones(self.dim_ee)
        self.eetgt = eetgt
        self.ee_idx = ee_idx
        self.jnt_idx = jnt_idx

        self.ramp_option = RAMP_CONSTANT
        self.l1 = 0.01
        self.l2 = 1.0
        self.alpha = 1e-5
        self.wu = 1e-3/np.array([3.09,1.08,0.393,0.674,0.111,0.152,0.098])  # Brett CostFK Big least squares

        ramp_len = self.ref_len if maxT is None else maxT
        self.wpm = get_ramp_multiplier(self.ramp_option, ramp_len, wp_final_multiplier=1.0)

    def eval(self, X, U, t, jac=None):
        # Constants.
        dX = X.shape[1]
        dU = U.shape[1]
        T = X.shape[0]

        wp = self.wp*np.expand_dims(self.wpm[t:t+T], axis=-1)

        l = 0.5 * np.sum(self.wu * (U ** 2), axis=1)
        lu = self.wu*U
        lx = np.zeros((T, dX))
        luu = np.tile(np.diag(self.wu), [T, 1, 1])
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        dist = X[:,self.ee_idx] - self.eetgt
        if len(jac.shape) == 3: # If jac has a time dimension
            Jd = jac[:,:,self.jnt_idx]
        else: # Rep single jacobian across time if not
            jac = jac[:, self.jnt_idx]
            Jd = np.tile(jac, [T, 1, 1])

        Jdd = np.zeros((T, self.dim_ee, self.dim_jnt, self.dim_jnt))
        l_fk, lx_fk, lxx_fk = evallogl2term( wp, dist, Jd, Jdd, self.l1, self.l2, self.alpha)

        #TODO: Add derivatives for the actual end-effector dimensions of state
        #Right now only derivatives w.r.t. joints are considered

        l += l_fk
        lx[:, self.jnt_idx] = lx_fk
        lxx[:, self.jnt_idx, self.jnt_idx] = lxx_fk

        return l, lx, lu, lxx, luu, lux

    def get_ee_tgt(self, t):
        """For RVIZ visualization"""
        return self.eetgt