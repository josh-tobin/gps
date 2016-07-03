import numpy as np
"""
This file contains default options for the MuJoCo configuration.
It should contain little executable code - just names and values.
"""

# General options
dX = 26
dU = 7
dT = dX+dU
jnt_idx = slice(0,7)
ee_idx = slice(14,20)
it = slice(0,33)
ip = slice(33,59)
H = 15 # Horizon
maxT = 100 # Total timesteps

# LQR Options
LQR_iter = 1  # Number of LQR iterations to take
min_mu = 1e-6  # LQR regularization
del0 = 2  # LQR regularization
lqr_discount = 0.9  # Discount factor.

# Noise
u_noise = 0.05

# Cost options
wu = 1e-2/np.array([3.09,1.08,0.593,0.674,0.111,0.152,0.098])  # Torque penalty
tgt = np.zeros(26)
eetgt = np.array([0.0, 0.3, -0.5,  0.0, 0.3, -0.2]) # End-effector target
tgt[ee_idx] = eetgt
use_jacobian = True

# Dynamics Options
dynamics_class = 'OnlineGaussianDynamics'
dyn_init_mu = np.zeros(dT+dX)
dyn_init_sig = np.eye(dT+dX)
prior_class = 'NoPrior'
prior_class_args = []
init_gamma = gamma = 0.1  # Higher means update faster. Lower means keep more history.
#init_gamma = gamma = 0.5
mix_prior_strength = 1.0

