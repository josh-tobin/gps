from __future__ import division

import numpy as np

from agent.mjc.agent_mjc import AgentMuJoCo
from sample_data.sample_data import SampleData


sd_hyperparams = {
    'T': 100,
    'dX': 55,
    'dU': 21,
    'dO': 55,
    'state_include': ['JointAngles', 'JointVelocities'],
    'obs_include': ['JointAngles', 'JointVelocities'],
    'state_idx': [list(range(28)), list(range(28,55))],
    'obs_idx': [list(range(28)), list(range(28,55))],
}

sample_data = SampleData(sd_hyperparams, {}, sample_writer=False)

agent_hyperparams = {
    'filename': '/home/marvin/dev/rlreloaded/domain_data/mujoco_worlds/humanoid.xml',
    'dt': 1/20,
}

agent = AgentMuJoCo(agent_hyperparams, sample_data)


class DummyPolicy(object):
    def __init__(self, dU):
        self.dU = dU

    def act(self, X, obs, t, noise):
        return np.ones(self.dU)

policy = DummyPolicy(21)

sample = agent.sample(policy, 100)
