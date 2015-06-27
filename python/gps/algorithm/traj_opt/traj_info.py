
class TrajectoryInfo(object):
    """
    Bundles information used by algorithm.
    Contains cost derivatives (cc, cv, Cm), dynamics
    """
    def __init__(self):
        self.dynamics = None
        self.x0mu = None
        self.x0sigma = None
        self.cc = None
        self.cv = None
        self.Cm = None