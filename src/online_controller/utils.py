import numpy as np
import scipy as sp
import scipy.linalg

def skew_3d(axis):
    """ Return skew-symmetric for cross product """
    return np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])

def axis_from_R(rot_mat):
    """
    Extracts rotation axis and magnitude from rot_mat
    Returns axis * theta.

    >>> theta = 2.0
    >>> axis = np.array([1,0,0])
    >>> rot_mat = sp.linalg.expm(theta*skew_3d(axis))
    >>> axis_angle_from_R(rot_mat)
    array([ 2.,  0.,  0.])
    """
    X = rot_mat-rot_mat.T
    theta = np.arccos(0.5*(np.trace(rot_mat)-1))
    s = np.array([X[2,1],X[0,2],X[1,0]])
    axis = s/(2*np.sin(theta))
    return axis*theta

