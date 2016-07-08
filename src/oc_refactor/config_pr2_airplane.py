import numpy as np
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY, \
                                          RAMP_CONSTANT, RAMP_QUADRATIC

dX = 32
dU = 7
eetgt = np.zeros(9)
ee_idx = slice(14, 23)
use_jacobian = True
H = 20
init_gamma = gamma = 0.01
u_noise = 0.01
LQR_iter = 1
lqr_discount = 0.9

# Cost options
wu = 3e-2/np.array([3.09, 1.08, 3.93, 0.674, 0.111, 0.152, 0.098])
l1 = 1.0 
l2 = 0.1
ramp_option = RAMP_QUADRATIC
wp = np.array([1.0, 1.0, 0.25, 1.0, 1.0, 0.25, 1.0, 1.0, 0.25])
final_penalty=1.0

use_offline_value = True
