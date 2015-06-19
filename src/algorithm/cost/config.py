"""Default configuration and hyperparameter values for cost objects

"""
from cost_utils import RAMP_CONSTANT, evallogl2term

""" CostFK """
cost_fk = {
    'ramp_option': RAMP_CONSTANT,  # How target cost increases over time.
    'wp': None,  # State weights - must be set
    'wp_final_multiplier': 1.0,  # Weight multiplier on final timestep
    'env_target': True,
    'analytic_jacobian': True,
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'evalnorm': evallogl2term
}

""" CostState """
# TODO - Desired state is also needed.
cost_state = {
    'ramp_option': RAMP_CONSTANT,  # How target cost increases over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final timestep
    'data_types': {
        'JointAngle': {
            'desired_state': None,  # Target state - must be set
            'wp': None  # State weights - must be set
        }
    }
}
