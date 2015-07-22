from traj_opt import TrajOpt

class TrajOptLQR(TrajOpt):
    """LQR trajectory optimization

    TODO: Implement this in C++, and use boost. This is just a placeholder.
    """
    def __init__(self, hyperparams, dynamics):
        TrajOpt.__init__(self, hyperparams, dynamics)
