"""
Default configuration and hyperparameter values for policy objects.
"""


# TODO - Make sure this is exposed at a higher level, also PD is default type.
""" Initial Linear Gaussian Trajectory Distributions """
init_lg = {
    # Used for both PD and LQR initialization.
    'init_var': 10.0,
    'init_stiffness': 10.0,
    'init_stiffness_vel': 0.01,
    # Used for LQR initialization only.
    'init_acc': [],  # dU vector of accelerations, zeros if not specified.
    'init_gains': [],  # dU vector of gains, ones if not specified.
    'init_final_weight': 1.0,
    # Used for PD initialization only, optional.
    'init_action_offset': None,
}


""" PolicyPrior """
policy_prior = {
    'strength': 1e-4,
}


""" PolicyPriorGMM """
policy_prior_gmm = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
    'keep_samples': True,
}
