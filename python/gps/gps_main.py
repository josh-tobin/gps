import logging

from hyperparam_defaults import defaults
from sample_data.sample_data import SampleData


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GPSMain():
    """Main class to run algorithms and experiments.

    Parameters
    ----------
    hyperparams: nested dictionary of hyperparameters, indexed by the type
        of hyperparameter
    """
    def __init__(self):
        self._hyperparams = defaults
        self._iterations = defaults['iterations']

        self.sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
        self.agent = defaults['agent']['type'](defaults['agent'], self.sample_data)
        defaults['algorithm']['init_traj_distr']['args']['x0'] = self.agent.x0
        self.algorithm = defaults['algorithm']['type'](defaults['algorithm'], self.sample_data)

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        for itr in range(self._iterations):
            # TODO - multiple times, for each condition
            sample = self.agent.sample(self.algorithm.cur[0].traj_distr, 100)
            sample_data.add_sample(sample)
            sample = self.agent.sample(self.algorithm.cur[0].traj_distr, 100)
            sample_data.add_sample(sample)
            n = self.sample_data.num_samples()-1
            self.algorithm.iteration([n-1, n-2])

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO")

g = GPSMain()
g.run()
