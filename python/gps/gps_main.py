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
        self._conditions = defaults['common']['conditions']

        self.sample_data = SampleData(defaults['sample_data'], defaults['common'], False)
        self.agent = defaults['agent']['type'](defaults['agent'], self.sample_data)
        # TODO: the following is a hack that doesn't even work some of the time
        #       let's think a bit about how we want to really do this
        defaults['algorithm']['init_traj_distr']['args']['x0'] = self.agent.x0[0]
        self.algorithm = defaults['algorithm']['type'](defaults['algorithm'], self.sample_data)

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        idxs = [[] for _ in range(self._conditions)]
        for itr in range(self._iterations):
            for m in range(self._conditions):
                for i in range(5):
                    n = self.sample_data.num_samples()
                    pol = self.algorithm.cur[m].traj_distr
                    sample = self.agent.sample(pol, self.sample_data.T, m, verbose=True)
                    self.sample_data.add_samples(sample)
                    idxs[m].append(n)
            self.algorithm.iteration([idx[-15:] for idx in idxs])

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO")

if __name__ == "__main__":
    g = GPSMain()
    g.run()
