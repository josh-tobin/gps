import numpy as np

dX = 32
dU = 7
eetgt = np.zeros(9)
ee_idx = slice(14, 23)
use_jacobian = True
H = 15
init_gamma = gamma = 0.05
u_noise = 0.005
LQR_iter = 1
use_offline_value = True
