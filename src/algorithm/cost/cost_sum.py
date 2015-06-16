from algorithm.cost.cost import Cost


class CostSum(Cost):
    """
    A wrapper cost function that adds other cost functions

    Args:
        costs: List of cost functions to add
        weights: List of weights, one for each cost function
    """

    def __init__(self, costs, weights):
        # super(self, CostSum).__init__(cost._hyperparams, cost.sample_data)
        self.costs = costs
        self.weights = weights

    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError()

    def eval(self, x, u, obs, sample_meta):
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
