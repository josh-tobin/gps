"""Default configuration and hyperparameter values for cost objects

"""

""" CostFK """
cost_fk = {
    'wu': 1e-4,  # Action penalty
    'wp': 1.0,  # Multiplier on each dimension of target deviation vector.
    'l1': 0.0,  # Multiplier for L1 component of cost.
    'l2': 1.0,  # Multiplier for L2 component of cost.
    'cost_ramp': 'quadratic',  # How target cost increases over time.
}

""" CostState """
# TODO - Desired state is also needed.
cost_state = {
    'wu': 1e-4,  # Action penalty
    # TODO - will the below wp work? Does it need to be 1xdX?
    'wp': 1.0,  # Multiplier on each dimension of target deviation vector.
    'l1': 1.0,  # Multiplier for L1 component of cost.
    'l2': 1.0,  # Multiplier for L2 component of cost.
    'alpha': 1e-2,  # Smoothing constant for smooth pseudo-L1 norm.
}
