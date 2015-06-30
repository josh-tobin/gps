import datetime
from __future__ import division

from agent.mjc.agent_mjc import AgentMuJoCo
from algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from algorithm.cost.cost_fk import CostFK
from algorithm.dynamics.dynamics_lr import DynamicsLR
from algorithm.traj_opt.traj_opt_lqr import TrajOptLQR
from algorithm.policy.lin_gauss_init import init_lqr

common = {
    'experiment_dir': 'experiments/default_experiment/',
    'experiment_name': 'my_experiment_'+datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%y_%H-%M'),
}

sample_data = {
    'filename': 'sample_data.pkl',
    'T': 100,
    'dX': 55,
    'dU': 21,
    'dO': 55,
    'state_include': ['JointAngles', 'JointVelocities'],
    #'state_include': ['JointAngles', 'JointVelocities', 'EndEffectorPoints', 'EndEffectorPointVelocities'],
    'obs_include': [],
    'state_idx': [list(range(28)), list(range(28,55))],
    'obs_idx': [list(range(28)), list(range(28,55))],
    #'obs_include': ['JointAngles', 'JointVelocities'],  # Input to policy
}

agent = {
    'type': AgentMuJoCo
    'filename': '/home/marvin/dev/rlreloaded/domain_data/mujoco_worlds/humanoid.xml',
    'dt': 1/20,
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': 1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'args': {
        'hyperparams': {},
        'dt': agent['dt']
    }
}

algorithm['cost'] = {
    'type': CostState,
    'data_types' : {
        'dummy': {
            'wp':np.ones((1,dX)),
            'desired_state': np.zeros((1,dX)),
            }
        },
}

algorithm['dynamics'] = {
    'type': DynamicsLR,
}

algorithm['traj_opt'] = {
        'type': TrajOptLQRPython,
        }

    #'type': TrajOptLQR,
    #TrajOptLQRPython({}),
#}

algorithm['policy_opt'] = {}

defaults = {
    'iterations': 10,
    'common': common,
    'sample': sample,
    'sample_data': sample_data,
    'agent': agent,
    'algorithm': algorithm,
}
