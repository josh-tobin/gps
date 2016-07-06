import numpy as np

from gps.algorithm.cost.cost_utils import get_ramp_multiplier, RAMP_CONSTANT, evall1l2term, evallogl2term, evallogl2term_fast

class CostFKOnline(object):
    """
    :param eetgt:
    """
    def __init__(self, eetgt, wu=None, jnt_tgt=None, jnt_wp=None, ee_idx=None, jnt_idx=None, use_jacobian=True, maxT=None):
        self.dim_ee = ee_idx.stop-ee_idx.start
        self.dim_jnt = jnt_idx.stop - jnt_idx.start
        self.wp = np.ones(self.dim_ee)
        #self.wp[0:3] = 3.0
        self.eetgt = eetgt
        self.ee_idx = ee_idx
        self.jnt_idx = jnt_idx
        self.jnt_tgt = jnt_tgt
        self.jnt_wp = jnt_wp
        self.use_jacobian = use_jacobian
        self.final_penalty = 2.0  # weight = sum of remaining weight * final penalty
        self.ramp_option = RAMP_CONSTANT
        self.l1 = 0.1
        self.l2 = 1.0
        self.alpha = 1e-5
        self.wu = wu

        ramp_len = self.ref_len if maxT is None else maxT
        self.wpm = get_ramp_multiplier(self.ramp_option, ramp_len, wp_final_multiplier=1.0)

    def eval(self, X, U, t, jac=None):
        # Constants.
        dX = X.shape[1]
        dU = U.shape[1]
        T = X.shape[0]

        wp = self.wp*np.expand_dims(self.wpm[t:t+T], axis=-1)
        remaining_weight = np.sum(self.wpm[t+T:])
        wp[-1,:] *= self.final_penalty

        l = 0.5 * np.sum(self.wu * (U ** 2), axis=1)
        lu = self.wu*U
        lx = np.zeros((T, dX))
        luu = np.tile(np.diag(self.wu), [T, 1, 1])
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        dist = X[:,self.ee_idx] - self.eetgt
	if self.use_jacobian:
            if len(jac.shape) == 3: # If jac has a time dimension
                Jd = jac[:,:,self.jnt_idx]
            else: # Rep single jacobian across time if not
                jac = jac[:, self.jnt_idx]
                Jd = np.tile(jac, [T, 1, 1])


        # Derivatives w.r.t. EE dimensions
        l_ee, lx_ee, lxx_ee = evallogl2term_fast(wp, dist, self.l1, self.l2, self.alpha)
        lx[:, self.ee_idx] = lx_ee
        lxx[:, self.ee_idx, self.ee_idx] = lxx_ee

        if self.jnt_tgt is not None:
            jwp = self.jnt_wp*np.expand_dims(self.wpm[t:t+T], axis=-1)
            jwp[-1,:] *= self.final_penalty
            jdist = X[:,self.jnt_idx] - self.jnt_tgt
            l_j, lx_j, lxx_j = evallogl2term_fast(jwp, jdist, self.l1, self.l2, self.alpha)
            lx[:, self.jnt_idx] += lx_j
            lxx[:, self.jnt_idx, self.jnt_idx] += lxx_j

        # Derivatives w.r.t. Joint dimensions
        #dist = dist[:,0:3]
        #Jd = Jd[:,0:3,:]
        #wp = wp[:,0:3]
	if self.use_jacobian:
            Jdd = np.zeros((T, self.dim_ee, self.dim_jnt, self.dim_jnt))
            l_fk, lx_fk, lxx_fk = evallogl2term( wp, dist, Jd, Jdd, self.l1, self.l2, self.alpha)
            l += l_fk
            lx[:, self.jnt_idx] += lx_fk
            lxx[:, self.jnt_idx, self.jnt_idx] += lxx_fk

        #TODO: Add derivatives for the actual end-effector dimensions of state
        #Right now only derivatives w.r.t. joints are considered
        print 'loss: %.2f'%np.sum(l)
	return l, lx, lu, lxx, luu, lux
