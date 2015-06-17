from cost import Cost


class CostSum(Cost):
    """
    A wrapper cost function that adds other cost functions
    """

    def __init__(self, hyperparams, sample_data):
        super(self, CostSum).__init__(hyperparams, sample_data)
        self.costs = hyperparams['costs']
        self.weights = hyperparams['weights']

    def eval(self, x, u, obs, sample_meta):
        """
        Evaluate cost function and derivatives

        Args:
            x: A T x Dx state matrix
            u: A T x Du action matrix
            obs: A T x Dobs observation matrix
            sample_meta: List of cost_info objects
                (temporary placeholder until we discuss how to pass these around)
        Return:
            l, lx, lu, lxx, luu, lux: Loss (Tx1 float) and 1st/2nd derivatives.
        """
        l, lx, lu, lxx, luu, lux = self.costs[0].eval(x, u, obs)
        l = l * self.weights[0]
        lx = lx * self.weights[0]
        lu = lu * self.weights[0]
        lxx = lxx * self.weights[0]
        luu = luu * self.weights[0]
        lux = lux * self.weights[0]
        for i in range(2, len(self.costs)):
            pl, plx, plu, plxx, pluu, plux = self.costs[0].eval(x, u, obs)
            l = l + pl * self.weights[i]
            lx = lx + plx * self.weights[i]
            lu = lu + plu * self.weights[i]
            lxx = lxx + plxx * self.weights[i]
            luu = luu + pluu * self.weights[i]
            lux = lux + plux * self.weights[i]
        return l, lx, lu, lxx, luu, lux
