

from gps.algorithm.policy.tf_policy  import TfPolicy

class LowDofPolicy(TfPolicy):
    def __init__(self, dU, obs_tensor, act_op, var, sess, device_string,
                 dofs,
                 hidden_state_tensor=None, initial_hidden_state_tensor=None,
                 tf_vars=None):
        super(LowDofPolicy, self).__init__(dU, obs_tensor, act_op, var, sess, 
                                       device_string, 
                                       hidden_state_tensor=hidden_state_tensor,
                                       initial_hidden_state_tensor=initial_hidden_state_tensor,
                                       tf_vars=None)
        self.dofs = dofs
