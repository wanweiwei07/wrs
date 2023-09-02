#!/usr/bin/env python3

import math
import numpy as np


def rodrigues(axis, theta):
    """
    Compute the rodrigues matrix using the given axis and theta

    ## input
    axis:
        a 1-by-3 numpy array list
    theta:
        angle in degree
    mat:
        a 3-by-3 numpy array, rotation matrix if this was not given,
        users could get it from return

    ## output
    the mat

    author: weiwei
    date: 20161220
    """

    theta = theta*math.pi/180.0
    axis = np.array([axis[0], axis[1], axis[2]])
    axis = axis/math.sqrt(np.dot(axis, axis))
    if theta > 2*math.pi:
        theta = theta % 2*math.pi
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2.0*(bc+ad), 2.0*(bd-ac)],
                     [2.0*(bc-ad), aa+cc-bb-dd, 2.0*(cd+ab)],
                     [2.0*(bd+ac), 2.0*(cd-ab), aa+dd-bb-cc]])


def hat(a):
    """
    compute the skew symmetric maxtix that corresponds to a cross

    :param a: a 1-by-3 numpy array
    :return: a 3-by-3 numpy array

    author: weiwei
    date: 20170421
    """

    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def rotmatfacet(facetnormal, facetfirstpoint, facetsecondpoint):
    '''
    Compute the rotation matrix of a 3D facet using
    facetnormal and the first two points on the facet
    The function uses the concepts defined by Trimesh

    ## input:
    facetnormal:
        the normal of a facet
    facetfirstpoint:
        the first point of the first triangle on the facet
    facetsecondpoint:
        the second point of the first triangle on the facet

    ## output:
    mat:
        a (3,3) numpy matrix

    date: 20160624
    author: weiwei
    '''

    mat = np.eye(3, 3)
    mat[2, :] = facetnormal
    mat[0, :] = facetsecondpoint-facetfirstpoint
    mat[0, :] = mat[0, :]/np.linalg.norm(mat[0, :])
    mat[1, :] = np.cross(mat[2, :], mat[0, :])

    return mat


def homoinverse(homomatrix4):
    """
    compute the inverse of a homogeneous transform

    :param homomatrix4:
    :return:

    author: weiwei
    date :20161213
    """

    rotmat = homomatrix4[:3, :3]
    tranvec = homomatrix4[:3, 3]
    invmatrix4 = np.eye(4, 4)
    invmatrix4[:3, :3] = np.transpose(rotmat)
    invmatrix4[:3, 3] = -np.dot(np.transpose(rotmat), tranvec)

    return invmatrix4


def transformmat4(matrix4, point):
    """
    do homotransform on point using matrix4

    :param matrix:
    :param point:
    :return:

    author: weiwei
    date: 20161213
    """

    point4 = np.array([point[0], point[1], point[2], 1])
    return np.dot(matrix4, point4)


def cvtRngPM180(armjnts):
    """
    change the range of armjnts to +-180

    :param armjnts a numpyarray of jnts
    date: 20170330
    author: weiwei
    """

    armjntsnew = armjnts.copy()
    for i in range(armjntsnew.shape[0]):
        if armjntsnew[i] < 0:
            armjntsnew[i] = armjntsnew[i] % -360
            if armjntsnew[i] < -180:
                armjntsnew[i] = armjntsnew[i] + 360
        if armjnts[i] > 0:
            armjntsnew[i] = armjntsnew[i] % 360
            if armjntsnew[i] > 180:
                armjntsnew[i] = armjntsnew[i] - 360

    return armjntsnew


def cvtRngPM360(armjnts):
    """
    change the range of armjnts to +-180

    :param armjnts a numpyarray of jnts
    date: 20170330
    author: weiwei
    """

    armjntsnew = armjnts.copy()
    for i in range(armjntsnew.shape[0]):
        if armjntsnew[i] < 0:
            armjntsnew[i] = armjntsnew[i] % -360
        if armjnts[i] > 0:
            armjntsnew[i] = armjntsnew[i] % 360

    return armjntsnew


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def radian_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def degree_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
