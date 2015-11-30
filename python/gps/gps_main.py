import logging

#from gps.hyperparam_badmm_defaults import defaults as config
from gps.hyperparam_defaults import defaults as config
#from gps.hyperparam_pr2 import defaults as config
#from gps.gui.gui import GUI


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GPSMain():
    """Main class to run algorithms and experiments.

    Parameters
    ----------
    hyperparams: nested dictionary of hyperparameters, indexed by the type
        of hyperparameter
    """
    def __init__(self):
        self._hyperparams = config
        self._iterations = config['iterations']
        self._conditions = config['common']['conditions']

        self.agent = config['agent']['type'](config['agent'])
        #if 'gui' in config:
        #    self.gui = GUI(self.agent, config['gui'])

        # TODO: the following is a hack that doesn't even work some of the time
        #       let's think a bit about how we want to really do this
        # TODO - the following line of code is needed for agent_mjc, but not agent_ros
        config['algorithm']['init_traj_distr']['args']['x0'] = self.agent.x0[0]
        config['algorithm']['init_traj_distr']['args']['dX'] = self.agent.dX
        config['algorithm']['init_traj_distr']['args']['dU'] = self.agent.dU
        config['algorithm']['dO'] = self.agent.dO

        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration step.
        """
        n = self._hyperparams['num_samples']
        for itr in range(self._iterations):
            for m in range(self._conditions):
                for i in range(n):
                    pol = self.algorithm.cur[m].traj_distr
                    self.agent.sample(pol, m, verbose=True)
            self.algorithm.iteration([self.agent.get_samples(m, -n) for m in range(self._conditions)])
            # Take samples from the policy to see how it is doing.
            #for m in range(self._conditions):
            #    self.agent.sample(self.algorithm.policy_opt.policy, m, verbose=True, save=False)

    def resume(self, itr):
        """
        Resume from iteration specified.
        """
        raise NotImplementedError("TODO")

if __name__ == "__main__":
    import random; random.seed(0)
    import numpy as np; np.random.seed(0)

    g = GPSMain()
    g.run()
