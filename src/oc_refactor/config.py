"""
This file contains default options for the MuJoCo configuration
"""
from online_dynamics import *
from cost_fk_online import CostFKOnline

# General options
dX = 26
dU = 7
dT = dX+dU
jnt_idx = slice(0,7)
ee_idx = slice(14,20)
it = slice(0,33)
ip = slice(33,59)
H = 20 # Horizon
maxT = 100 # Total timesteps

# LQR Options
LQR_iter = 1  # Number of LQR iterations to take
min_mu = 1e-6  # LQR regularization
del0 = 2  # LQR regularization
lqr_discount = 0.9  # Discount factor.

# Noise
u_noise = 0.03

# Cost options
wu = 3e-3/np.array([3.09,1.08,0.593,0.674,0.111,0.152,0.098])  # Torque penalty
tgt = np.zeros(26)
eetgt = np.array([0.0, 0.3, -0.5,  0.0, 0.3, -0.2]) # End-effector target
tgt[ee_idx] = eetgt
cost = CostFKOnline(eetgt, wu=wu, ee_idx=ee_idx, jnt_idx=jnt_idx, maxT=maxT, use_jacobian=True)

# Dynamics Options
dyn_init_mu = np.zeros(dT+dX)
dyn_init_sig = np.eye(dT+dX)
prior = NoPrior()
init_gamma = gamma = 0.1  # Higher means update faster. Lower means keep more history.
dynamics = OnlineDynamics(gamma, prior, dyn_init_mu, dyn_init_sig)
mix_prior_strength = 1.0

