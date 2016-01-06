from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM


""" DynamicsLRPrior """
dyn_lr_prior = {
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
    },
}


""" DynamicsPriorGMM """
dyn_prior_gmm = {
    'min_samples_per_cluster': 20,
    'max_clusters': 50,
    'max_samples': 20,
    'strength': 1.0,
}
