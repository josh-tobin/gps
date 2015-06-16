"""Default configuration and hyperparameter values for cost objects

"""

""" CostFK """
cost_fk = {
    'cost_ramp': 'quadratic',  # How target cost increases over time.
}

""" CostState """
cost_state = {
    'cost_ramp': 'constant',  # How the target cost increases over time
    'velocity_ramp': 'same',  # How the velocity cost increases over time
}
