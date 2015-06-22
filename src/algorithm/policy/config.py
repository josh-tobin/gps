"""Default configuration and hyperparameter values for algorithm objects

"""

# TODO - might want to just have a single dictionary for all init types?
# TODO - make sure this is exposed at a higher level, and merged together in code.
""" Intial Linear Gaussian Traj Distributions """
init_pd = {
    'init_var': 10.0,
    'init_stiffness',10.0,
    'init_stiffness_vel',0.01,
    'init_gains',1.0,
    'init_acc',0.0,
    'init_action_offset',NaN,
    'init_final_weight',1.0,
    'init_polwt',0.01,
}
