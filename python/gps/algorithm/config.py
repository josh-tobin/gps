"""Default configuration and hyperparameter values for algorithm objects

"""

""" AlgorithmTrajOpt """
alg_traj_opt = {
    'inner_iterations': 4,  # Number of iterations
    'min_eta': 1e-5,  # minimum initial lagrange multiplier in DGD for trajopt
    # Step size settings
    'kl_step',0.2,
    'min_step_mult',0.1,
    'max_step_mult',10.0,
    'sample_decrease_var',0.5,
    'sample_increase_var',1.0,
    # Trajectory settings.
    'initial_state_var',1e-6,
}
