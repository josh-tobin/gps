import datetime

from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from algorithm.cost.cost_fk import CostFK
from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.traj_opt.traj_opt_lqr import TrajOptLQR

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
    'state_include': ['JointAngles', 'JointVelocities', 'EndEffectorPoints', 'EndEffectorPointVelocities'],
    'obs_include': ['JointAngles', 'JointVelocities'],  # Input to policy
}

agent = {}

algorithm = {
    'type': AlgorithmTrajOpt,
}

algorithm['cost'] = {
    'type': CostFK,
}

algorithm['dynamics'] = {
    'type': DynamicsLR,
}

algorithm['traj_opt'] = {
    'type': TrajOptLQR,
}

algorithm['policy_opt'] = {}

defaults = {
    'iterations': 10,
    'common': common,
    'sample': sample,
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}
