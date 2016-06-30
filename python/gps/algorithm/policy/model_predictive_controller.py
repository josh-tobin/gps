from gps.algorithm.policy.policy import Policy
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
import numpy as np

class ModelPredictiveController(Policy):

    def __init__(self, algorithm, condition, smoothing_factor=0.0, initial_eta=1.0, step_mult=1.0):
        self.algorithm = algorithm
        self.initial_lqr = algorithm.cur[condition].traj_distr
        self.prev_traj_distr = self.initial_lqr
        self.traj_info = algorithm.cur[condition].traj_info
        self.condition = condition
        self.initial_eta = initial_eta
        self.prev_eta = initial_eta
        self.step_mult = step_mult
        self.mu_0 = self.traj_info.x0mu
        self.last_kl_step_init = self.traj_info.last_kl_step
        self.reset()
    
    def calc_adjustment(self):
        if self.time_index == 0:
            return np.zeros_like(self.error_history[0,:])
        else:
            return (( self.smoothing_factor * 
                      self.error_history[self.time_index-1, :]) + 
                    ((1-self.smoothing_factor ) * 
                     self.error_history[self.time_index,:]
                    )
                   )
    def reset(self):
        self.previous_prediction = np.zeros([self.algorithm.dX,])
        self.prev_traj_distr = self.initial_lqr
        self.error_history = np.zeros([self.algorithm.T, self.algorithm.dX])
        self.traj_info.x0mu = self.mu_0
        self.time_index = 0
        self.algorithm._eval_cost(self.condition)
        self.traj_info.last_kl_step = self.last_kl_step_init    
    #def iteration(self, prev_traj_distr, traj_info, prev_eta, step_mult, 
    #              algorithm, m, T):
    def act(self, x, obs, t, noise):
        if t == 0:
            err = np.zeros_like(x)
        else:
            err = x - self.previous_prediction
        
        self.error_history[self.time_index, :] = err  
        self.traj_info.x0mu = x
        
        # Update dynamics function
        # TBD
        old_dynamics = self.traj_info.dynamics.dynamics_function
        self.traj_info.dynamics.dynamics_function = lambda xu: old_dynamics(xu) + self.calc_adjustment()

        # Fit new dynamics
        new_traj_distr, new_eta = \
                self.algorithm.traj_opt.iteration(
                        self.prev_traj_distr, 
                        self.traj_info,
                        self.prev_eta, 
                        self.step_mult, 
                        self.algorithm, 
                        self.condition,
                        self.algorithm.T - self.time_index
                )
        u = new_traj_distr.act(x, obs, 0, noise)
        self.traj_info.dynamics.dynamics_function = old_dynamics
        self.previous_prediction = self.traj_info.dynamics.dynamics_function(
            np.concatenate([x, u])
        )

        self.prev_traj_distr = LinearGaussianPolicy(
                new_traj_distr.K[1:,:,:],
                new_traj_distr.k[1:,:], 
                new_traj_distr.pol_covar[1:,:,:],
                new_traj_distr.chol_pol_covar[1:,:,:],
                new_traj_distr.inv_pol_covar[1:,:,:]
        )
        self.time_index += 1

        return u 
