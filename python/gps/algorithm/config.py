"""Default configuration and hyperparameter values for algorithm objects

"""
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.dynamics.dynamics_lr import DynamicsLR

""" AlgorithmTrajOpt """
alg_traj_opt = {
    'inner_iterations': 1,  # Number of iterations
    'min_eta': 1e-5,  # minimum initial lagrange multiplier in DGD for trajopt
    # Step size settings
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    'sample_decrease_var':0.5,
    'sample_increase_var':1.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy objects for each condition
    # TrajOpt
    'traj_opt': TrajOptLQRPython({}),
    # Dynamics hyperaparams
    'dynamics': {
        'type': DynamicsLR
    },
    # Costs
    'cost': None,  # A list of Cost objects for each condition
    # TODO: Remove these:
    'conditions': 0
}


""" AlgorithmBADMM """
alg_badmm = {
    'inner_iterations': 1,  # Number of iterations
    'min_eta': 1e-5,  # minimum initial lagrange multiplier in DGD for trajopt
    # Step size settings
    'kl_step':0.2,
    'min_step_mult':0.01,
    'max_step_mult':10.0,
    'sample_decrease_var':0.5,
    'sample_increase_var':1.0,
    # Trajectory settings.
    'initial_state_var':1e-6,
    'init_traj_distr': None,  # A list of initial LinearGaussianPolicy objects for each condition
    # TrajOpt
    'traj_opt': TrajOptLQRPython({}),
    # Dynamics hyperaparams
    'dynamics': {
        'type': DynamicsLR
    },
    # Costs
    'cost': None,  # A list of Cost objects for each condition

    'policy_dual_rate': 1.0,
    'policy_dual_rate_covar': 0.0,
    'fixed_lg_step': 0,
    'lg_step_schedule': 10.0,
    'ent_reg_schedule': 0.0,
}
