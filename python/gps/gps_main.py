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
        idxs = []
        for itr in range(self._iterations):
            # TODO - multiple times, for each condition
            for i in range(10):
                n = self.sample_data.num_samples()
                sample = self.agent.sample(self.algorithm.cur[0].traj_distr, self.sample_data.T)
                self.sample_data.add_samples(sample)
                idxs.append(n)
            self.algorithm.iteration([idxs[-30:]])

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO")

g = GPSMain()
g.run()
