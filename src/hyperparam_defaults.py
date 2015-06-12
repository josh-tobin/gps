import datetime

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(),'%m-%d-%y_%H-%M'),
}

sample = {
    'state_include': ['JointAngle', 'JointVelocity', 'EndEffectorPose', 'EndEffectorVelocity'],
    'obs_include': [], # Input to policy
}

agent = {}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
}

cost = {}

dynamics = {}

traj_opt = {}

policy_opt = {}

defaults = {
    'common': common,
    'sample': sample,
    'agent': agent,
    'sample_data': sample_data,
    'algorithm': {
        'cost': cost,
        'dynamics': dynamics,
        'traj_opt': traj_opt,
        'policy_opt': policy_opt,
    },
}
