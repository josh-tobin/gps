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

        self.agent = defaults['agent']['type'](defaults['agent'])
        # TODO: the following is a hack that doesn't even work some of the time
        #       let's think a bit about how we want to really do this
        defaults['algorithm']['init_traj_distr']['args']['x0'] = self.agent.x0[0]
        defaults['algorithm']['init_traj_distr']['args']['dX'] = self.agent.dX
        defaults['algorithm']['init_traj_distr']['args']['dU'] = self.agent.dU
        self.algorithm = defaults['algorithm']['type'](defaults['algorithm'])

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        for itr in range(self._iterations):
            for m in range(self._conditions):
                for i in range(5):
                    pol = self.algorithm.cur[m].traj_distr
                    self.agent.sample(pol, m, verbose=True)
            self.algorithm.iteration([self.agent.get_samples(-5) for _ in range(self._conditions)])

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO")

if __name__ == "__main__":
    g = GPSMain()
    g.run()
