#!/usr/bin/env python

class GPSMain():
    """Main class to run algorithms and experiments.

    Parameters
    ----------
    hyperparams: nested dictionary of hyperparameters, indexed by the type
        of hyperparameter
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._iterations = hyperparams['algorithm']['iterations']

        # TODO - keep track of experiment name, sample data, etc.

        # TODO - need to make sure these initialize the right type of agent
        # and algorithm objects: either change code to use switch statement or
        # put in respective constructors, or do something fancier.
        self.sample_data = SampleData();
        self.algorithm = Algorithm(hyperparams['algorithm'], sample_data);
        self.agent = Agent(hyperparams['agent'], sample_data);

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        for itr in range(self._iterations):
            self.agent.sample()
            self.algorithm.iteration()

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO");
