""" Default configuration and hyperparameter values for policies. """


# TODO - Make sure this is exposed at a higher level, also PD is default
#        type.
# Initial Linear Gaussian Trajectory Distributions
INIT_LG = {
    # Used for both PD and LQR initialization.
    'init_var': 10.0,
    'init_stiffness': 10.0,
    'init_stiffness_vel': 0.01,
    # Used for LQR initialization only.
    'init_acc': [],  # dU vector of accelerations, default zeros.
    'init_gains': [],  # dU vector of gains, default ones.
    'init_final_weight': 1.0,
    # Used for PD initialization only, optional.
    'init_action_offset': None,
}


# PolicyPrior
POLICY_PRIOR = {
    'strength': 1e-4,
}


# PolicyPriorGMM
POLICY_PRIOR_GMM = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
    'keep_samples': True,
}
