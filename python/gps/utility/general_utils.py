def bundletype(name, vars):
    """
    Creates a class that bundles many fields.
    This is similar to a record or mutable namedtuple

    Args:
        name: Name of class to create
        vars: A list of string representing field names.
            A field with the name _frozen cannot be used (it is used internally).
    Returns:
        A class object. All fields default to None

    >>> bundle_class = bundletype('MyBundle', ['a', 'b', 'c'])
    >>> bundle1 = bundle_class()
    >>> bundle1.a = 5
    >>> bundle1.d = 4
    Traceback (most recent call last):
    ...
    AttributeError: MyBundle{'a': 5, 'c': None, 'b': None} has no attribute d
    """
    if '_frozen' in vars:
        raise ValueError('Invalid field name _frozen')

    class BundleType(object):
        def __init__(self):
            for var in vars:
                setattr(self, var, None)
            self._frozen = True  # Freeze fields

        def __repr__(self):
            return name+str({var: getattr(self, var) for var in vars})

    BundleType.__name__ = name

    # Freeze fields so new ones cannot be set.
    def __setattr(self, key, value):
        if hasattr(self, '_frozen') and not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)
    BundleType.__setattr__ = __setattr
    return BundleType

ITERATION_VARS = ['sample_list', 'traj_info', 'pol_info', 'traj_distr', 'cs',
                  'step_change', 'pol_kl', 'step_mult', 'mispred_std']
IterationData = bundletype('IterationData', ITERATION_VARS)

TRAJINFO_VARS = ['dynamics', 'x0mu', 'x0sigma', 'cc', 'cv', 'Cm', 'last_kl_step']
TrajectoryInfo = bundletype('TrajectoryInfo', TRAJINFO_VARS)

POLINFO_VARS = ['lambda_k', 'lambda_K', 'pol_wt', 'pol_mu', 'pol_sig',
                'pol_K', 'pol_k', 'pol_S', 'chol_pol_S', 'prev_kl']
PolicyInfo = bundletype('PolicyInfo', POLINFO_VARS)
