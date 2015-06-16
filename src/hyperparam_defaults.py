import datetime

from algorithm.lqr_alg import LQRAlgorithm
from algorithm.cost.cost_fk import CostFK
from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.traj_opt.traj_opt_lqr import TrajOptLQR

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample = {
    'state_include': ['JointAngle', 'JointVelocity', 'EndEffectorPose', 'EndEffectorVelocity'],
    'obs_include': [],  # Input to policy
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
}

agent = {}

algorithm = {
    'type': LQRAlgorithm,
    'iterations': 10,
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
    'common': common,
    'sample': sample,
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}
