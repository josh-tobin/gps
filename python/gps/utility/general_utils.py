
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

def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape

    Args:
        value: Matrix to shape check
        expected_shape: A tuple or list of integers
        name (optional): An optional name to add to the exception message.
            Default empty string.
    Returns: None
    >>> import numpy as np
    >>> a = np.zeros((4, 6))
    >>> check_shape(a, (4,6))
    >>> check_shape(a, (4,), name='a')
    Traceback (most recent call last):
    ...
    ValueError: Shape mismatch a: Expected (4,), got (4, 6)
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
            (name, str(expected_shape), str(value.shape)))

