"""Default configuration and hyperparameter values for algorithm objects

"""

# TODO - make sure this is exposed at a higher level.
""" Intial Linear Gaussian Traj Distributions """
init_lg = {
    # Used for both pd and lqr initialization
    'init_var': 10.0,
    'init_stiffness': 10.0,
    'init_stiffness_vel': 0.01,
    # Used for lqr initialization only
    'init_acc': [],  # numpy vector of dU accelerations, ones if not specified
    'init_gains': [],  # numpy vector of dU gains, ones if not specified
    'init_final_weight': 1.0,
    # Used for pd initialization only, optional
    'init_action_offset',NaN,
}
