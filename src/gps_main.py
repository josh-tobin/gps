from hyperparams_defaults import defaults

class GPSMain():
    """Main class to run algorithms and experiments.

    Parameters
    ----------
    hyperparams: nested dictionary of hyperparameters, indexed by the type
        of hyperparameter
    """
    def __init__(self):
        self._hyperparams = defaults
        self._iterations = defaults['algorithm']['iterations']

        self.sample_data = SampleData(defaults['sample_data'], defaults['common'])
        self.algorithm = defaults['algorithm']['type'](defaults['algorithm'], defaults['common'], sample_data)
        self.agent = defaults['agent']['type'](defaults['agent'], defaults['common'], sample_data)

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
        raise NotImplementedError("TODO")
