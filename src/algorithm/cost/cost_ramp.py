class CostRamp(Cost):
    """
    A wrapper cost function that applies a ramping multiplier over time

    Args:
        cost: Cost function to ramp
        ramp_option: 'constant', 'linear', 'quadratic', or 'final_only'
    """
    def __init__(self, cost, ramp_option):
        super(self, CostRamp).__init__(cost._hyperparams, cost.sample_data)
        self.cost = cost
        self.ramp_option = ramp_option

    def update(self):
        """ Update cost values and derivatives. """
        raise NotImplementedError();

