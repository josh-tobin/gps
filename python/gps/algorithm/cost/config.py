"""Default configuration and hyperparameter values for cost objects

"""
import numpy as np

from gps.algorithm.cost.cost_utils import RAMP_CONSTANT, evallogl2term


""" CostFK """
cost_fk = {
    'ramp_option': RAMP_CONSTANT,  # How target cost increases over time.
    'wp': None,  # State weights - must be set
    'wp_final_multiplier': 1.0,  # Weight multiplier on final timestep
    'env_target': True, # TODO - this isn't used
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-5,
    'target_end_effector': None,  # Target end-effector position
    'evalnorm': evallogl2term
}

""" CostState """
cost_state = {
    'ramp_option': RAMP_CONSTANT,  # How target cost increases over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final timestep
    'data_types': {
        'JointAngle': {
            'target_state': None,  # Target state - must be set
            'wp': None  # State weights - must be set
        }
    }
}

""" CostSum """
# Below is the structure of the hyperparams - just a template, code not used.
cost_sum = {
    'costs': [],  # A list of hyperparam dictionaries for each cost
    'weights': [],  # Weight multipliers for each cost
}

""" CostTorque """
cost_torque = {
    'wu': np.array([]),  # Torque penalties, must be 1 x dU numpy array
}
