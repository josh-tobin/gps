#!/usr/bin/env python
import numpy as np

from hyperparam_defaults import *
from sample_data.sample_data import SampleData
from agent.dummy_agent import DummyAgent
from algorithm.cost.cost_state import CostState

class Main():
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._iterations = hyperparams['algorithm']['iterations']

        # put in respective constructors, or do something fancier.
        state_assembler = None #StateAssembler(hyperparams['state'])
        self.sample_data = SampleData(hyperparams['sample_data'], hyperparams['common'], state_assembler);
        #self.algorithm = Algorithm(hyperparams['algorithm'], hyperparams['common'], sample_data, state_assembler);

        tmp_agent_hyperparams = {'T': 100, 'dX': 8, 'dU': 3, 'dPhi': 2}
        self.agent = DummyAgent(tmp_agent_hyperparams, hyperparams['common'], self.sample_data, state_assembler);

        tgt = np.array([1,1,1,1,1,1,1,1])
        wp = np.array([1,1,1,1,0,0,0,0])
        self.cost = CostState(hyperparams[ALGORITHM][COST], self.sample_data, tgt, wp)

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        for itr in range(self._iterations):
            self.agent.sample(3)
            self.cost.update()
            #self.algorithm.iteration()

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO");

def main():
    print "Starting"
    hyperparams = defaults
    hyperparams['algorithm']['iterations'] = 5
    m = Main(hyperparams)
    m.run()

if __name__ == "__main__":
    main()