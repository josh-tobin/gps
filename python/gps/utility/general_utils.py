
import numpy as np
import xml.etree.ElementTree as ElementTree
import multiprocessing as mp
#import pathos.multiprocessing as mp
from functools import partial

class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable
    namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


def check_shape(value, expected_shape, name=''):
    """
    Throws a ValueError if value.shape != expected_shape.
    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.
    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' %
                         (name, str(expected_shape), str(value.shape)))


def finite_differences(func, inputs, func_output_shape=(), epsilon=1e-5):
    """
    Computes gradients via finite differences.
    derivative = (func(x+epsilon) - func(x-epsilon)) / (2*epsilon)
    Args:
        func: Function to compute gradient of. Inputs and outputs can be
            arbitrary dimension.
        inputs: Vector value to compute gradient at.
        func_output_shape: Shape of the output of func. Default is
            empty-tuple, which works for scalar-valued functions.
        epsilon: Difference to use for computing gradient.
    Returns:
        Gradient vector of each dimension of func with respect to each
        dimension of input.
    """
    gradient = np.zeros(inputs.shape+func_output_shape)
    for idx, _ in np.ndenumerate(inputs):
        test_input = np.copy(inputs)
        test_input[idx] += epsilon
        obj_d1 = func(test_input)
        assert obj_d1.shape == func_output_shape
        test_input = np.copy(inputs)
        test_input[idx] -= epsilon
        obj_d2 = func(test_input)
        assert obj_d2.shape == func_output_shape
        diff = (obj_d1 - obj_d2) / (2 * epsilon)
        gradient[idx] += diff
    return gradient

"""
class ParallelFiniteDifferences(object):
    ''' Class that stores parallel workers to compute finite differences on several
        threads '''
    def __init__(self, func, func_output_shape=(), epsilon=1e-5, n_workers=6):
        self.func = func
        self.func_output_shape = func_output_shape
        self.epsilon=epsilon
        
        self.map_func = partial(finite_differences, self.func, 
                func_output_shape=self.func_output_shape, epsilon=self.epsilon)
        self.pool = mp.Pool(n_workers)

    #def map_func(self, inputs):
    #    return finite_differences(self.func, inputs, 
    #            func_output_shape=self.func_output_shape, epsilon=self.epsilon)

    def __call__(self, inputs):
        def map_func(inputs):
            return finite_differences(self.func, inputs, 
                func_output_shape=self.func_output_shape, epsilon=self.epsilon)
        result = self.pool.map(map_func, inputs)
        result = np.stack(result)
        return result
"""

class ParallelFiniteDifferences(object):
    def __init__(self, n_workers=6):
        self.n_workers=n_workers
        self.pool = mp.Pool(n_workers)

    def __call__(self, func, inputs, func_output_shape=(), epsilon=1e-5):
        def map_func(x):
            return finite_differences(func, x, func_output_shape, epsilon)
        result = self.pool.map(map_func, inputs)
        return np.stack(result)

    def __getstate__(self):
        return {'n_workers': self.n_workers}
    def __setstate__(self, state):
        self.__init__(state['n_workers'])

def approx_equal(a, b, threshold=1e-5):
    """
    Return whether two numbers are equal within an absolute threshold.
    Returns:
        True if a and b are equal within threshold.
    """
    return np.all(np.abs(a - b) < threshold)


def extract_condition(hyperparams, m):
    """
    Pull the relevant hyperparameters corresponding to the specified
    condition, and return a new hyperparameter dictionary.
    """
    return {var: val[m] if isinstance(val, list) else val
            for var, val in hyperparams.items()}


def get_ee_points(offsets, ee_pos, ee_rot):
    """
    Helper method for computing the end effector points given a
    position, rotation matrix, and offsets for each of the ee points.

    Args:
        offsets: N x 3 array where N is the number of points.
        ee_pos: 1 x 3 array of the end effector position.
        ee_rot: 3 x 3 rotation matrix of the end effector.
    Returns:
        3 x N array of end effector points.
    """
    return ee_rot.dot(offsets.T) + ee_pos.T

def find_objects_in_model(model_path):
    """
    Helper function to find all of the non-robot objects in the scene
    so that we can apply gravity to them individually (as MJC does not
    seem to have a way to selectively apply gravity to only some bodies).
    
    Args:
        model_path: the path to the xml model file
    Returns:
        objects: dictionary mapping object names to their position in the
                 object array, which can be used by the agent to access
                 them.
    """
    root = ElementTree.parse(model_path).getroot()
    objects = {}
    # keep track of the position of the objects
    object_pos = 0 
    for body in root.iter('body'):
        # First, we increment object pos, to account for the fact
        # that tree.iter('body') misses one body in the model: namely,
        # worldbody

        object_pos += 1
        if 'name' in body.attrib:
            # We use the convention that objects will be identified in the
            # xml file by appending OBJ to their names.
            if body.attrib['name'][:3] == 'OBJ':
                objects[body.attrib['name']] = object_pos
    return objects

def find_sites_in_model(model_path):
    root = ElementTree.parse(model_path).getroot()
    sites = {}
    site_pos = 0
    for site in root.iter('site'):
        if 'name' in site.attrib:
            sites[site.attrib['name']] = site_pos
        site_pos += 1
    return sites

def find_bodies_in_model(model_path, list_of_bodies):
    """
    Helper function to find all of the links with given names in the 
    scene so we can adjust their masses, etc.
    """
    root = ElementTree.parse(model_path).getroot()
    objects = {}
    # keep track of position of the objects
    object_pos = 0
    for body in root.iter('body'):
        object_pos += 1
        if 'name' in body.attrib:
            if body.attrib['name'] in list_of_bodies:
                objects[body.attrib['name']] = object_pos
    return objects
