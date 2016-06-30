from gps.algorithm.dynamics.dynamics import Dynamics
from gps.utility.general_utils import finite_differences
import numpy as np

class DynamicsTrue(Dynamics):
    """ Encodes the true dynamics (of a simulator) and computes its 
        linearization. """
    def __init__(self, hyperparams):
        Dynamics.__init__(self, hyperparams)
        self.Fm = None
        self.fv = None
        self.dyn_covar = None
        #self.world = hyperparams['world']
        #self.dX_include = hyperparams['dX_include']
        #self.substeps = hyperparams['substeps']
        self.agent = hyperparams['agent']
        self.condition = hyperparams['condition']
    def update_prior(self, sample):
        """ Update dynamics prior. """
        # Nothing to do
        pass

    def get_prior(self):
        """ Return the dynamics prior, or None if constant prior """
        return None

    def fit(self, sample_list):
        """ Fit dynamics. In this case, we will simply linearize the 
            dynamics around the (mean) reference trajectory """
        X = sample_list.get_X()
        U = sample_list.get_U()
        if len(X.shape) == 2:
            N = 1
            self.T, self.dX = X.shape
            self.dU = U.shape[1]
            mean_X = X
            mean_U = U
        else:
            self.N, self.T, self.dX = X.shape
            self.dU = U.shape[2]
            
            mean_X = np.mean(X, axis=0)
            mean_U = np.mean(U, axis=0)
        self.Fm = np.zeros([self.T, self.dX, self.dX+self.dU])
        self.fv = np.zeros([self.T, self.dX])
        self.dyn_covar = np.zeros([self.T, self.dX, self.dX])
        
        if hasattr(self.agent, 'dX_model'):
            self.dX_include = self.agent.dX_model 
        else: 
            self.dX_include = self.dX

        xu = np.c_[mean_X, mean_U]
        errs = np.zeros([self.T,])
        for t in range(self.T-1):
            df_dxu = finite_differences(self.dynamics_function, xu[t,:], (self.dX,),
                                        epsilon=1e-5).T
            
            self.Fm[t,:,:] = df_dxu
            self.fv[t,:] = xu[t+1,:self.dX] - df_dxu.dot(xu[t,:])
            self.dyn_covar[t,:,:] = 1e-6 * np.eye(self.dX, self.dX)


    def dynamics_function(self, xu):
        x = xu[:self.dX_include]
        u = xu[self.dX:self.dX+self.dU]
        new_x = self.agent.step(self.condition, x, u)
            
        if self.dX_include < self.dX:
            prev_eepts = xu[self.dX_include:self.dX]
            idx = len(prev_eepts) // 2 # Only want pos, not vel
            prev_eepts = prev_eepts[:idx]
            eepts, eepts_vel, eepts_jac = self.agent.get_ee_obs(prev_eepts,
                                                            self.condition)
            new_x = np.concatenate([new_x, eepts, eepts_vel])
        return new_x

