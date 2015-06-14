class CostSum(Cost):
    """
    A wrapper cost function that adds other cost functions

    Args:
        costs: List of cost functions to add
        weights: List of weights, one for each cost function
    """
    def __init__(self, costs, weights):
        #TODO: Which hyperparameter dictionary to choose???
        super(self, CostSum).__init__(cost._hyperparams, cost.sample_data)
        self.costs = costs
        self.weights = weights

    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError("Must be implemented in subclass");
