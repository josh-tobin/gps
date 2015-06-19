from cost import Cost


class CostSum(Cost):
    """
    A wrapper cost function that adds other cost functions
    """
    def __init__(self, hyperparams):

        assert len(hyperparams['costs']) == len(hyperparams['weights'])

        self._costs = []

        for cost in hyperparams['costs']:
            self._costs.append(cost['type'](cost['hyperparams']))


    def eval(self, sample):
        """
        Evaluate cost function and derivatives

        Args:
            sample: A Sample object
        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample)
        weights = self._hyperparams['weights'][0]
        l = l * weights
        lx = lx * weights
        lu = lu * weights
        lxx = lxx * weights
        luu = luu * weights
        lux = lux * weights
        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample)
            weights = self._hyperparams['weights'][i]
            l = l + pl * weights
            lx = lx + plx * weights
            lu = lu + plu * weights
            lxx = lxx + plxx * weights
            luu = luu + pluu * weights
            lux = lux + plux * weights
        return l, lx, lu, lxx, luu, lux
