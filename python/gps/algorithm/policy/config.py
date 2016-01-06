"""Default configuration and hyperparameter values for algorithm objects

"""

# TODO - make sure this is exposed at a higher level, also pd is default type.
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
