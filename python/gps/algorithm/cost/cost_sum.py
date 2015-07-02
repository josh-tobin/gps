from cost import Cost


class CostSum(Cost):
    """
    A wrapper cost function that adds other cost functions
    """
    def __init__(self, hyperparams):

        assert len(hyperparams['costs']) == len(hyperparams['weights'])

        self._costs = []
        self._weights = hyperparams['weights']

        for cost in hyperparams['costs']:
            self._costs.append(cost['type'](cost['hyperparams']))


    def eval(self, sample_idx):
        """
        Evaluate cost function and derivatives

        Args:
            sample_idx:  A single index into sample_data
        Return:
            l, lx, lu, lxx, luu, lux:
                Loss (len T float) and derivatives with respect to states (x) and/or actions (u).
        """
        l, lx, lu, lxx, luu, lux = self._costs[0].eval(sample_idx)

        # Compute weighted sum of each evaluated cost value and derivative
        weight = self._weights[0]
        l = l * weight
        lx = lx * weight
        lu = lu * weight
        lxx = lxx * weight
        luu = luu * weight
        lux = lux * weight
        for i in range(1, len(self._costs)):
            pl, plx, plu, plxx, pluu, plux = self._costs[i].eval(sample_idx)
            weight = self._weights[i]
            l = l + pl * weight
            lx = lx + plx * weight
            lu = lu + plu * weight
            lxx = lxx + plxx * weight
            luu = luu + pluu * weight
            lux = lux + plux * weight
        return l, lx, lu, lxx, luu, lux
