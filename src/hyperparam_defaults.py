import datetime

COMMON = 'common'
SAMPLE = 'sample'
AGENT = 'agent'
SAMPLE_DATA = 'sample_data'
ALGORITHM = 'algorithm'
COST = 'cost'

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample = {
    'state_include': ['JointAngle', 'JointVelocity', 'EndEffectorPose', 'EndEffectorVelocity'],
    'obs_include': [],  # Input to policy
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
    COMMON: common,
    SAMPLE: sample,
    AGENT: agent,
    SAMPLE_DATA: sample_data,
    ALGORITHM: {
        COST: cost,
        'dynamics': dynamics,
        'traj_opt': traj_opt,
        'policy_opt': policy_opt,
    },
}
