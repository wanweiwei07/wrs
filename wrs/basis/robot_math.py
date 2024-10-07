import math
import scipy
import operator
import warnings
import functools
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import wrs.basis.constant as const
import wrs.basis.trimesh.creation as trm_creation
from sklearn import cluster
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
from numpy import radians, degrees, sign, zeros, eye, pi, sqrt
from numpy.linalg import norm
from numpy import sin, cos, tan
from numpy import arctan2 as atan2, arcsin as asin, arccos as acos
from numpy import floor, ceil, round

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(np.float32).eps
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# helpers
def vec(*args):
    return np.array(args)


## rotmat
def rotmat_from_axangle(axis, angle):
    """
    Compute the rodrigues matrix using the given axis and angle

    :param axis: 1x3 nparray
    :param angle:  angle in radian
    :return: 3x3 rotmat
    author: weiwei
    date: 20161220
    """
    axis = unit_vector(axis)
    if np.allclose(axis, np.zeros(3)):
        return np.eye(3)
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2.0 * (bc + ad), 2.0 * (bd - ac)],
                     [2.0 * (bc - ad), aa + cc - bb - dd, 2.0 * (cd + ab)],
                     [2.0 * (bd + ac), 2.0 * (cd - ab), aa + dd - bb - cc]])


def rotmat_from_quaternion(quaternion):
    """
    convert a quaterion to rotmat
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def rotmat_to_quaternion(rotmat):
    """
    convert a rotmat to quaternion
    :param rotmat:
    :return:
    """
    return Rotation.from_matrix(rotmat).as_quat()


def rotmat_to_wvec(rotmat):
    """
    convert a rotmat to angle*ax form
    :param rotmat:
    :return:
    """
    return Rotation.from_matrix(rotmat).as_rotvec()


def rotmat_from_normal(surfacenormal):
    '''
    Compute the rotation matrix of a 3D mesh using a surface normal
    :param surfacenormal: 1x3 nparray
    :return: 3x3 rotmat
    date: 20160624
    author: weiwei
    '''
    rotmat = np.eye(3, 3)
    rotmat[:, 2] = unit_vector(surfacenormal)
    rotmat[:, 0] = orthogonal_vector(rotmat[:, 2], toggle_unit=True)
    rotmat[:, 1] = np.cross(rotmat[:, 2], rotmat[:, 0])
    return rotmat


def rotmat_from_normalandpoints(facetnormal, facetfirstpoint, facetsecondpoint):
    '''
    Compute the rotation matrix of a 3D facet using
    facet_normal and the first two points on the facet
    The function uses the concepts defined by Trimesh
    :param facetnormal: 1x3 nparray
    :param facetfirstpoint: 1x3 nparray
    :param facetsecondpoint: 1x3 nparray
    :return: 3x3 rotmat
    date: 20160624
    author: weiwei
    '''
    rotmat = np.eye(3, 3)
    rotmat[:, 2] = unit_vector(facetnormal)
    rotmat[:, 0] = unit_vector(facetsecondpoint - facetfirstpoint)
    if np.allclose(rotmat[:, 0], 0):
        warnings.warn("The provided facetpoints are the same! An autocomputed vector is used instead...")
        rotmat[:, 0] = orthogonal_vector(rotmat[:, 2], toggle_unit=True)
    rotmat[:, 1] = np.cross(rotmat[:, 2], rotmat[:, 0])
    return rotmat


def rotmat_from_euler(ai, aj, ak, order='sxyz'):
    """
    :param ai: radian
    :param aj: radian
    :param ak: radian
    :param order:
    :return:
    author: weiwei
    date: 20190504
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[order]
    except (AttributeError, KeyError):
        _TUPLE2AXES[order]  # validation
        firstaxis, parity, repetition, frame = order
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    rotmat = np.eye(3)
    if repetition:
        rotmat[i, i] = cj
        rotmat[i, j] = sj * si
        rotmat[i, k] = sj * ci
        rotmat[j, i] = sj * sk
        rotmat[j, j] = -cj * ss + cc
        rotmat[j, k] = -cj * cs - sc
        rotmat[k, i] = -sj * ck
        rotmat[k, j] = cj * sc + cs
        rotmat[k, k] = cj * cc - ss
    else:
        rotmat[i, i] = cj * ck
        rotmat[i, j] = sj * sc - cs
        rotmat[i, k] = sj * cc + ss
        rotmat[j, i] = cj * sk
        rotmat[j, j] = sj * ss + cc
        rotmat[j, k] = sj * cs - sc
        rotmat[k, i] = -sj
        rotmat[k, j] = cj * si
        rotmat[k, k] = cj * ci
    return rotmat


def rotmat_to_euler(rotmat, order='sxyz'):
    """
    :param rotmat: 3x3 nparray
    :param order: order
    :return: radian
    author: weiwei
    date: 20190504
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[order.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[order]  # validation
        firstaxis, parity, repetition, frame = order
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if repetition:
        sy = math.sqrt(rotmat[i, j] * rotmat[i, j] + rotmat[i, k] * rotmat[i, k])
        if sy > _EPS:
            ax = math.atan2(rotmat[i, j], rotmat[i, k])
            ay = math.atan2(sy, rotmat[i, i])
            az = math.atan2(rotmat[j, i], -rotmat[k, i])
        else:
            ax = math.atan2(-rotmat[j, k], rotmat[j, j])
            ay = math.atan2(sy, rotmat[i, i])
            az = 0.0
    else:
        cy = math.sqrt(rotmat[i, i] * rotmat[i, i] + rotmat[j, i] * rotmat[j, i])
        if cy > _EPS:
            ax = math.atan2(rotmat[k, j], rotmat[k, k])
            ay = math.atan2(-rotmat[k, i], cy)
            az = math.atan2(rotmat[j, i], rotmat[i, i])
        else:
            ax = math.atan2(-rotmat[j, k], rotmat[j, j])
            ay = math.atan2(-rotmat[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return np.array([ax, ay, az])


def rotmat_between_vectors(v1, v2):
    """
    from v1 to v2?
    :param v1: 1-by-3 nparray
    :param v2: 1-by-3 nparray
    :return:
    author: weiwei
    date: 20191228
    """
    theta = angle_between_vectors(v1, v2)
    if np.allclose(theta, 0):
        return np.eye(3)
    if np.allclose(theta, np.pi):  # in this case, the rotation axis is arbitrary; I am using v1 for reference
        return rotmat_from_axangle(orthogonal_vector(v1, toggle_unit=True), theta)
    axis = unit_vector(np.cross(v1, v2))
    return rotmat_from_axangle(axis, theta)


def rotmat_average(rotmatlist, bandwidth=10):
    """
    average a list of rotmat (3x3)
    :param rotmatlist:
    :param denoise: meanshift denoising is applied if True
    :return:
    author: weiwei
    date: 20190422
    """
    if len(rotmatlist) == 0:
        return False
    quaternion_list = []
    for rotmat in rotmatlist:
        quaternion_list.append(quaternion_from_rotmat(rotmat))
    quat_avg = quaternion_average(quaternion_list, bandwidth=bandwidth)
    rotmat_avg = rotmat_from_quaternion(quat_avg)
    return rotmat_avg


def rotmat_slerp(rotmat0, rotmat1, nval):
    """
    :param rotmat0:
    :param rotmat1:
    :param nval:
    :return: 1xnval list of slerped rotmat including rotmat0 and rotmat1
    """
    key_rots = R.from_matrix((rotmat0, rotmat1))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    slerp_times = np.linspace(key_times[0], key_times[1], nval)
    interp_rots = slerp(slerp_times)
    return interp_rots.as_matrix()


## homogeneous matrix
def homomat_from_posrot(pos=np.zeros(3), rotmat=np.eye(3)):
    """
    build a 4x4 nparray homogeneous matrix
    :param pos: nparray 1x3
    :param rotmat: nparray 3x3
    :return:
    author: weiwei
    date: 20190313
    """
    homomat = np.eye(4)
    homomat[:3, :3] = rotmat
    homomat[:3, 3] = pos
    return homomat


def homomat_from_pos_axanglevec(pos=np.zeros(3), axangle=np.ones(3)):
    """
    build a 4x4 nparray homogeneous matrix
    :param pos: nparray 1x3
    :param axanglevec: nparray 1x3, correspondent unit vector is rotation motion_vec; axis_length is radian rotation angle
    :return:
    author: weiwei
    date: 20200408
    """
    ax, angle = unit_vector(axangle, toggle_length=True)
    rotmat = rotmat_from_axangle(ax, angle)
    return homomat_from_posrot(pos, rotmat)


def homomat_inverse(homomat):
    """
    compute the inverse of a homogeneous transform
    :param homomat: 4x4 homogeneous matrix
    :return:
    author: weiwei
    date :20161213
    """
    rotmat = homomat[:3, :3]
    pos = homomat[:3, 3]
    inv_homomat = np.eye(4)
    inv_homomat[:3, :3] = np.transpose(rotmat)
    inv_homomat[:3, 3] = -np.dot(np.transpose(rotmat), pos)
    return inv_homomat


def homomat_average(homomat_list, bandwidth=10):
    """
    average a list of pos (4x4)
    :param homomat_list:
    :param bandwidth:
    :param denoise:
    :return:
    author: weiwei
    date: 20200109
    """
    homomat_array = np.asarray(homomat_list)
    pos_avg = pos_average(homomat_array[:, :3, 3], bandwidth)
    rotmat_avg = rotmat_average(homomat_array[:, :3, :3], bandwidth)
    return homomat_from_posrot(pos_avg, rotmat_avg)


def homomat_from_quaternion(quaternion):
    """
    convert a quaterion to homomat
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray):
    """
    do homotransform on a point or an array of points using pos
    :param homomat:
    :param points: 1x3 nparray or nx3 nparray
    :return:
    author: weiwei
    date: 20161213
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be np.ndarray!")
    if points.ndim == 1:
        homo_point = np.insert(points, 3, 1)
        return np.dot(homomat, homo_point)[:3]
    else:
        homo_points = np.ones((4, points.shape[0]))
        homo_points[:3, :] = points.T[:3, :]
        transformed_points = np.dot(homomat, homo_points).T
        return transformed_points[:, :3]


def interplate_pos_rotmat(start_pos,
                          start_rotmat,
                          goal_pos,
                          goal_rotmat,
                          granularity=.01):
    """
    :param start_info: [pos, rotmat]
    :param goal_info: [pos, rotmat]
    :param granularity
    :return: a list of 1xn nparray
    """
    len, vec = unit_vector(start_pos - goal_pos, toggle_length=True)
    n_steps = math.ceil(len / granularity)
    if n_steps == 0:
        n_steps = 1
    pos_list = np.linspace(start_pos, goal_pos, n_steps)
    rotmat_list = rotmat_slerp(start_rotmat, goal_rotmat, n_steps)
    return zip(pos_list, rotmat_list)


def interplate_pos_rotmat_around_circle(circle_center_pos,
                                        circle_normal_ax,
                                        radius,
                                        start_rotmat,
                                        end_rotmat,
                                        granularity=.01):
    """
    :param circle_center_pos:
    :param start_rotmat:
    :param end_rotmat:
    :param granularity: meter between two key points in the workspace
    :return:
    """
    vec = orthogonal_vector(circle_normal_ax)
    angular_step_length = granularity / radius
    n_angular_steps = math.ceil(np.pi * 2 / angular_step_length)
    rotmat_list = rotmat_slerp(start_rotmat, end_rotmat, n_angular_steps)
    pos_list = []
    for angle in np.linspace(0, np.pi * 2, n_angular_steps).tolist():
        pos_list.append(np.dot(rotmat_from_axangle(circle_normal_ax, angle), vec * radius) + circle_center_pos)
    return zip(pos_list, rotmat_list)


def interpolate_vectors(start_vector, end_vector, granularity):
    """
    :param start_vector:
    :param end_vector:
    :param num_points:
    :param include_ends:
    :return:
    """
    max_diff = np.max(np.abs(end_vector - start_vector))
    num_intervals = np.ceil(max_diff / granularity).astype(int)
    num_points = num_intervals + 1
    interpolated_vectors = np.linspace(start_vector, end_vector, num_points)
    return interpolated_vectors


# quaternion
def quaternion_from_axangle(angle, axis):
    """
    :param angle: radian
    :param axis: 1x3 nparray
    author: weiwei
    date: 20201113
    """
    quaternion = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen, _ = unit_vector(quaternion, toggle_length=True)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[0] = math.cos(angle / 2.0)
    return quaternion


def quaternion_average(quaternion_list, bandwidth=10):
    """
    average a list of quaternion (nx4)
    this is the full version
    :param rotmatlist:
    :param bandwidth: meanshift denoising is applied if available
    :return:
    author: weiwei
    date: 20190422
    """
    if len(quaternion_list) == 0:
        return False
    quaternionarray = np.array(quaternion_list)
    if bandwidth is not None:
        anglelist = []
        for quaternion in quaternion_list:
            anglelist.append([quaternion_to_axangle(quaternion)[0]])
        mt = cluster.MeanShift(bandwidth=bandwidth)
        quaternionarray = quaternionarray[np.where(mt.fit(anglelist).labels_ == 0)]
    nquat = quaternionarray.shape[0]
    weights = [1.0 / nquat] * nquat
    # Form the symmetric accumulator matrix
    accummat = np.zeros((4, 4))
    wsum = 0
    for i in range(nquat):
        q = quaternionarray[i, :]
        w_i = weights[i]
        accummat += w_i * (np.outer(q, q))  # rank 1 update
        wsum += w_i
    # scale
    accummat /= wsum
    # Get the eigenvector corresponding to largest eigen value
    quatavg = np.linalg.eigh(accummat)[1][:, -1]
    return quatavg


def quaternion_to_euler(quaternion, order='sxyz'):
    """
    :param rotmat: 3x3 nparray
    :param order: order
    :return: radian
    author: weiwei
    date: 20190504
    """
    return rotmat_to_euler(rotmat_from_quaternion(quaternion), order)


def quaternion_from_euler(ai, aj, ak, order='sxyz'):
    """
    Return quaternion from Euler angles and axis sequence.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[order.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[order]  # validation
        firstaxis, parity, repetition, frame = order
    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1
    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    q = np.empty((4,))
    if repetition:
        q[0] = cj * (cc - ss)
        q[i] = cj * (cs + sc)
        q[j] = sj * (cc + ss)
        q[k] = sj * (cs - sc)
    else:
        q[0] = cj * cc + sj * ss
        q[i] = cj * sc - sj * cs
        q[j] = cj * ss + sj * cc
        q[k] = cj * cs - sj * sc
    if parity:
        q[j] *= -1.0
    return q


def quaternion_about_axis(angle, axis):
    """
    Return quaternion for rotation about axis.
    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen, _ = unit_vector(q, toggle_length=True)
    if qlen > _EPS:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)
    return q


def quaternion_to_rotmat(quaternion):
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def quaternion_from_rotmat(rotmat):
    """
    return quaternion from rotation matrix
    """
    q = np.empty((4,))
    t = np.trace(rotmat)
    q[0] = t
    q[3] = rotmat[1, 0] - rotmat[0, 1]
    q[2] = rotmat[0, 2] - rotmat[2, 0]
    q[1] = rotmat[2, 1] - rotmat[1, 2]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions.
    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_conjugate(quaternion):
    """
    Return conjugate of quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """
    Return inverse of quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def quaternion_real(quaternion):
    """
    Return real part of quaternion.
    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """
    Return imaginary part of quaternion.
    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * np.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def rand_quaternion():
    """
    Return uniform random unit quaternion.
    :return:
    """
    rand = np.random.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2) * r2, np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2])


def rand_rotmat():
    """
    Return uniform random rotation matrix.
    """
    return quaternion_to_rotmat(rand_quaternion())


def skew_symmetric(posvec):
    """
    compute the skew symmetric maxtix that corresponds to a cross
    :param posvec: 1x3 nparray
    :return: 3x3 skew symmetric matrix
    author: weiwei
    date: 20170421
    """
    return np.array([[0, -posvec[2], posvec[1]],
                     [posvec[2], 0, -posvec[0]],
                     [-posvec[1], posvec[0], 0]])


def orthogonal_vector(npvec3, toggle_unit=True):
    """
    given a vector np.array([a,b,c]),
    this function computes an orthogonal one using np.array([b-c, -a+c, a-c])
    and then make it unit
    :param npvec3: 1x3 nparray
    :return: a 1x3 unit nparray
    author: weiwei
    date: 20200528
    """
    a = npvec3[0]
    b = npvec3[1]
    c = npvec3[2]
    if toggle_unit:
        return unit_vector(np.array([b - c, -a + c, a - b]))
    else:
        return np.array([b - c, -a + c, a - b])


def rel_pose(pos0, rotmat0, pos1, rotmat1):
    """
    relpos of rot1, pos1 with respect to rot0 pos0
    :param rot0: 3x3 nparray
    :param pos0: 1x3 nparray
    :param rot1:
    :param pos1:
    :return:
    author: weiwei
    date: 20180811, 20240223
    """
    rel_pos = rotmat0.T @ (pos1 - pos0)
    rel_rotmat = rotmat0.T @ rotmat1
    return (rel_pos, rel_rotmat)


def regulate_angle(lowerbound, upperbound, jntangles):
    """
    change the range of armjnts to [lowerbound, upperbound]
    NOTE: upperbound-lowerbound must be multiplies of 2*np.pi or 360
    :param lowerbound
    :param upperbound
    :param jntangles: an array or a single joint angle
    :return:
    """
    if isinstance(jntangles, np.ndarray):
        rng = upperbound - lowerbound
        if rng >= 2 * np.pi:
            jntangles[jntangles < lowerbound] = jntangles[jntangles < lowerbound] % -rng + rng
            jntangles[jntangles > upperbound] = jntangles[jntangles > upperbound] % rng - rng
        else:
            raise ValueError("upperbound-lowerbound must be multiplies of 2*np.pi or 360")
        return jntangles
    else:
        rng = upperbound - lowerbound
        if rng >= 2 * np.pi:
            jntangles = jntangles % -rng + rng if jntangles < lowerbound else jntangles % rng - rng
        else:
            raise ValueError("upperbound-lowerbound must be multiplies of 2*np.pi or 360")
        return jntangles


def unit_vector(vector, toggle_length=False):
    """
    :param vector: 1-by-3 nparray
    :return: the unit of a vector
    author: weiwei
    date: 20200701osaka
    """
    length = np.linalg.norm(vector)
    if math.isclose(length, 0):
        if toggle_length:
            return 0.0, np.zeros_like(vector)
        else:
            return np.zeros_like(vector)
    if toggle_length:
        return length, vector / np.linalg.norm(vector)
    else:
        return vector / np.linalg.norm(vector)


def angle_between_vectors(v1, v2):
    """
    :param v1: 1-by-3 nparray
    :param v2: 1-by-3 nparray
    :return:
    author: weiwei
    date: 20190504
    """
    l1, v1_u = unit_vector(v1, toggle_length=True)
    l2, v2_u = unit_vector(v2, toggle_length=True)
    if l1 == 0 or l2 == 0:
        raise ValueError("Zero length vector!")
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_2d_vecs(v1, v2):
    """
    return the angle from v1 to v2, with signs
    :param v1: 2d vector
    :param v2:
    :return:
    author: weiwei
    date: 20210530
    """
    return math.atan2(v2[1] * v1[0] - v2[0] * v1[1], v2[0] * v1[0] + v2[1] * v1[1])


# def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
#     """
#     compute angle*ax from src_rotmat to tgt_rotmat
#     the following relation holds for the returned delta_w
#     rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
#     :param src_rotmat: 3x3 nparray
#     :param tgt_rotmat: 3x3 nparray
#     :return:
#     author: weiwei
#     date: 20200326
#     """
#     delta_rotmat = tgt_rotmat @ src_rotmat.T
#     tmp_vec = np.array([delta_rotmat[2, 1] - delta_rotmat[1, 2],
#                         delta_rotmat[0, 2] - delta_rotmat[2, 0],
#                         delta_rotmat[1, 0] - delta_rotmat[0, 1]])
#     tmp_vec_norm = np.linalg.norm(tmp_vec)
#     if tmp_vec_norm > _EPS:
#         delta_w = np.arctan2(tmp_vec_norm, np.trace(delta_rotmat) - 1.0) / tmp_vec_norm * tmp_vec
#     elif delta_rotmat[0, 0] > 0 and delta_rotmat[1, 1] > 0 and delta_rotmat[2, 2] > 0:
#         delta_w = np.array([0, 0, 0])
#     else:
#         delta_w = np.pi / 2.0 * (np.diag(delta_rotmat) + 1)
#     return delta_w


def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
    """
    compute angle*ax from src_rotmat to tgt_rotmat
    the following relation holds for the returned delta_w
    rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
    :param src_rotmat: 3x3 nparray
    :param tgt_rotmat: 3x3 nparray
    :return:
    author: weiwei
    date: 20240416
    """
    delta_rotmat = tgt_rotmat @ src_rotmat.T
    return Rotation.from_matrix(delta_rotmat).as_rotvec()
    # tmp_vec = np.array([delta_rotmat[2, 1] - delta_rotmat[1, 2],
    #                     delta_rotmat[0, 2] - delta_rotmat[2, 0],
    #                     delta_rotmat[1, 0] - delta_rotmat[0, 1]])
    # tmp_vec_norm = np.linalg.norm(tmp_vec)
    # tmp_trace_minus_one = np.trace(delta_rotmat) - 1.0
    # if np.isclose(tmp_vec_norm, 0.0):
    #     return np.zeros(3)
    # elif np.isclose(tmp_trace_minus_one, 0.0):
    #     return np.pi / 2.0 * np.diag(delta_rotmat)
    # else:
    #     return np.arctan2(tmp_vec_norm, tmp_trace_minus_one) / tmp_vec_norm * tmp_vec


# def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
#     """
#     compute angle*ax from src_rotmat to tgt_rotmat
#     the following relation holds for the returned delta_w
#     rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
#     :param src_rotmat: 3x3
#     :param tgt_rotmat: 3x3
#     :return:
#     author: weiwei
#     date: 20200326
#     """
#     delta_rotmat = tgt_rotmat @ src_rotmat.T
#     clipped_trace = np.clip((np.trace(delta_rotmat) - 1) / 2.0, -1.0, 1.0)
#     angle = np.arccos(clipped_trace)
#     if np.isclose(angle, 0.0):
#         return np.zeros(3)
#     else:
#         axis = np.array([delta_rotmat[2, 1] - delta_rotmat[1, 2],
#                          delta_rotmat[0, 2] - delta_rotmat[2, 0],
#                          delta_rotmat[1, 0] - delta_rotmat[0, 1]]) / (2 * np.sin(angle))
#         return angle * axis


def diff_between_poses(src_pos,
                       src_rotmat,
                       tgt_pos,
                       tgt_rotmat):
    """
    compute the error between the given tcp and tgt_tcp
    :param src_pos:
    :param src_rotmat
    :param tgt_pos: the position vector of the goal (could be a single value or a list of jntid)
    :param tgt_rotmat: the rotation matrix of the goal (could be a single value or a list of jntid)
    :return: a 1x6 nparray where the first three indicates the displacement in pos,
                the second three indictes the displacement in rotmat
    author: weiwei
    date: 20230929
    """
    delta = np.zeros(6)
    delta[0:3] = (tgt_pos - src_pos)
    delta[3:6] = delta_w_between_rotmat(src_rotmat, tgt_rotmat)
    pos_err = np.linalg.norm(delta[:3])
    rot_err = np.linalg.norm(delta[3:6])
    return pos_err, rot_err, delta


def cosine_between_vecs(v1, v2):
    l1, v1_u = unit_vector(v1, toggle_length=True)
    l2, v2_u = unit_vector(v2, toggle_length=True)
    if l1 == 0 or l2 == 0:
        raise Exception("One of the given vector is [0,0,0].")
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def axangle_between_rotmat(rotmati, rotmatj):
    deltaw = delta_w_between_rotmat(rotmati, rotmatj)
    angle = np.linalg.norm(deltaw)
    ax = deltaw / angle if isinstance(deltaw, np.ndarray) else None
    return ax, angle


def quaternion_to_axangle(quaternion):
    """
    :param quaternion:
    :return: angle (radian), axis
    author: weiwei
    date: 20190421
    """
    lim = 1e-12
    norm = np.linalg.norm(quaternion)
    angle = 0
    axis = [0, 0, 0]
    if norm > lim:
        w = quaternion[0]
        vec = quaternion[1:]
        normvec = np.linalg.norm(vec)
        angle = 2 * math.acos(w)
        axis = vec / normvec
    return angle, axis


def pos_average(pos_list, bandwidth=10):
    """
    average a list of posvec (1x3)
    :param pos_list:
    :param denoise: meanshift denoising is applied if True
    :return:
    author: weiwei
    date: 20190422
    """
    if len(pos_list) == 0:
        return False
    if bandwidth is not None:
        mt = cluster.MeanShift(bandwidth=bandwidth)
        pos_avg = mt.fit(pos_list).cluster_centers_[0]
        return pos_avg
    else:
        return np.array(pos_list).mean(axis=0)


def gen_icorotmats(icolevel=1,
                   rotation_interval=math.radians(45),
                   crop_normal=np.array([0, 0, 1]),
                   crop_angle=np.pi,
                   toggle_flat=False):
    """
    generate rotmats using icospheres and rotationaangle each origin-vertex vector of the icosphere
    :param icolevel, the default value 1 = 42vertices
    :param rotation_interval
    :param crop_normal: crop results around a normal with crop_angle (crop out a cone section)
    :param crop_angle:
    :return: [[rotmat3, ...], ...] size of the inner list is size of the angles
    author: weiwei
    date: 20191015osaka
    """
    return_list = []
    icos = trm_creation.icosphere(icolevel)
    for vert in icos.vertices:
        if crop_angle < np.pi:
            if angle_between_vectors(vert, crop_normal) > crop_angle:
                continue
        z = -vert
        x = orthogonal_vector(z)
        y = unit_vector(np.cross(z, x))
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        return_list.append([])
        for angle in np.linspace(0, 2 * np.pi, int(2 * np.pi / rotation_interval), endpoint=False):
            return_list[-1].append(np.dot(rotmat_from_axangle(z, angle), temprotmat))
    if toggle_flat:
        return functools.reduce(operator.iconcat, return_list, [])
    return return_list


def gen_icohomomats(icolevel=1,
                    position=np.array([0, 0, 0]),
                    rotation_interval=math.radians(45),
                    toggle_flat=False):
    """
    generate homomats using icospheres and rotationaangle each origin-vertex vector of the icosphere
    :param icolevel, the default value 1 = 42vertices
    :param rot_angles, 8 directions by default
    :return: [[pos, ...], ...] size of the inner list is size of the angles
    author: weiwei
    date: 20200701osaka
    """
    rot_angles = np.linspace(strat=0, stop=2 * np.pi, num=np.pi * 2 / rotation_interval, endpoint=False)
    returnlist = []
    icos = trm_creation.icosphere(icolevel)
    for vert in icos.vertices:
        z = -vert
        x = orthogonal_vector(z)
        y = unit_vector(np.cross(z, x))
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        returnlist.append([])
        for angle in rot_angles:
            tmphomomat = np.eye(4)
            tmphomomat[:3, :3] = np.dot(rotmat_from_axangle(z, angle), temprotmat)
            tmphomomat[:3, 3] = position
            returnlist[-1].append(tmphomomat)
    if toggle_flat:
        return functools.reduce(operator.iconcat, returnlist, [])
    return returnlist


def gen_2d_spiral_points(max_radius: float = .002,
                         radial_granularity: float = .0001,
                         tangential_granularity: float = .0003,
                         toggle_origin: bool = False) -> npt.NDArray:
    """
    gen spiral curve
    :param max_radius:
    :param radial_granularity:
    :param tangential_granularity:
    :param toggle_origin: include 0 or not
    :return:
    """
    # if tangential_granularity > radial_granularity * np.pi:
    #     warnings.warn("The tangential_granularity is suggested to be smaller than 3*radial_granularity!")
    r = np.arange(radial_granularity, max_radius, radial_granularity)
    t_ele = tangential_granularity / r
    t = np.cumsum(t_ele)
    x = r * np.cos(t)
    y = r * np.sin(t)
    if toggle_origin:
        x.insert(0, 0)
        y.insert(0, 0)
    return np.column_stack((x, y))


def gen_3d_spiral_points(pos: npt.NDArray = np.zeros(3),
                         rotmat: npt.NDArray = np.eye(3),
                         max_radius: float = .002,
                         radial_granularity: float = .0001,
                         tangential_granularity: float = .0003,
                         toggle_origin: bool = False) -> npt.NDArray:
    """
    gen spiral curve
    :param pos
    :param rotmat
    :param max_radius:
    :param tangential_granularity:
    :param toggle_origin: include 0 or not
    :return:
    """
    xy_spiral_points = gen_2d_spiral_points(max_radius=max_radius,
                                            radial_granularity=radial_granularity,
                                            tangential_granularity=tangential_granularity,
                                            toggle_origin=toggle_origin)
    xyz_spiral_points = np.column_stack((xy_spiral_points, np.zeros(len(xy_spiral_points))))
    return rotmat.dot(xyz_spiral_points.T).T + pos


def gen_regpoly(radius, nedges=12):
    angle_list = np.linspace(0, np.pi * 2, nedges + 1, endpoint=True)
    x_vertex = np.sin(angle_list) * radius
    y_vertex = np.cos(angle_list) * radius
    return np.column_stack((x_vertex, y_vertex))


def gen_2d_isosceles_verts(nlevel, edge_length, nedges=12):
    xy_array = np.asarray([[0, 0]])
    for level in range(nlevel):
        xy_vertex = gen_regpoly(radius=edge_length * (level + 1), nedges=nedges)
        for i in range(nedges):
            xy_array = np.append(xy_array,
                                 np.linspace(xy_vertex[i, :], xy_vertex[i + 1, :], num=level + 1, endpoint=False),
                                 axis=0)
    return xy_array


def gen_2d_equilateral_verts(nlevel, edge_length):
    return gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=6)


def gen_3d_isosceles_verts(pos, rotmat, nlevel=5, edge_length=0.001, nedges=12):
    xy_array = gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=nedges)
    xyz_array = np.pad(xy_array, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    return rotmat.dot((xyz_array).T).T + pos


def gen_3d_equilateral_verts(pos, rotmat, nlevel=5, edge_length=0.001):
    return gen_3d_isosceles_verts(pos=pos, rotmat=rotmat, nlevel=nlevel, edge_length=edge_length, nedges=6)


def get_aabb(pointsarray):
    """
    get the axis aligned bounding box of nx3 array
    :param pointsarray: nx3 array
    :return: center + np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
    author: weiwei
    date: 20191229
    """
    xmax = np.max(pointsarray[:, 0])
    xmin = np.min(pointsarray[:, 0])
    ymax = np.max(pointsarray[:, 1])
    ymin = np.min(pointsarray[:, 1])
    zmax = np.max(pointsarray[:, 2])
    zmin = np.min(pointsarray[:, 2])
    center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
    # volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    return [center, np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])]


def compute_pca(nparray):
    """
    :param nparray: nxd array, d is the dimension
    :return: evs eigenvalues, axes_mat dxn array, each column is an eigenvector
    author: weiwei
    date: 20200701osaka
    """
    ca = np.cov(nparray, y=None, rowvar=False, bias=True)  # rowvar row=point, bias biased covariance
    pcv, pcaxmat = np.linalg.eig(ca)
    return pcv, pcaxmat


def transform_data_pcv(data, random_rot=True):
    """
    :param data:
    :param random_rot:
    :return:
    author: reuishuang
    date: 20210706
    """
    pcv, pcaxmat = compute_pca(data)
    inx = sorted(range(len(pcv)), key=lambda k: pcv[k])
    x_v = pcaxmat[:, inx[2]]
    y_v = pcaxmat[:, inx[1]]
    z_v = pcaxmat[:, inx[0]]
    pcaxmat = np.asarray([y_v, x_v, -z_v]).T
    if random_rot:
        pcaxmat = np.dot(rotmat_from_axangle([1, 0, 0], math.radians(5)), pcaxmat)
        pcaxmat = np.dot(rotmat_from_axangle([0, 1, 0], math.radians(5)), pcaxmat)
        pcaxmat = np.dot(rotmat_from_axangle([0, 0, 1], math.radians(5)), pcaxmat)
    transformed_data = np.dot(pcaxmat.T, data.T).T
    return transformed_data, pcaxmat


def fit_plane(points):
    """
    :param points: nx3 nparray
    :return:
    """
    plane_center = points.mean(axis=0)
    result = np.linalg.svd(points - plane_center)
    plane_normal = unit_vector(np.cross(result[2][0], result[2][1]))
    return plane_center, plane_normal


def project_point_to_plane(point, plane_center, plane_normal):
    dist = abs((point - plane_center).dot(plane_normal))
    # print((point - plane_center).dot(plane_normal))
    if (point - plane_center).dot(plane_normal) < 0:
        plane_normal = - plane_normal
    projected_point = point - dist * plane_normal
    return projected_point


def project_vector_to_vector(vector1, vector2):
    return (vector1 @ vector2) * vector2 / (vector2 @ vector2)


def distance_point_to_edge(point, edge_start, edge_end):
    """
    compute the minimum distance from a point to a line segment.
    :param point: ndarray of the point coordinates.
    :param edge_start: ndarray of the starting point of the segment.
    :param edge_end: ndarray of the end point of the segment.
    :return: minimum distance from the point to the line segment, and the projection point
    """
    edge_vector = edge_end - edge_start
    point_vector = point - edge_start
    segment_length_squared = np.dot(edge_vector, edge_vector)
    if segment_length_squared == 0:
        return np.linalg.norm(point_vector)
    t = max(0, min(1, np.dot(point_vector, edge_vector) / segment_length_squared))
    projection = edge_start + t * edge_vector
    return np.linalg.norm(point - projection), projection


def min_distance_point_edge_list(contact_point, edge_list):
    """
    compute the minimum distance between a point and a list of nested edge lists.
    :param contact_point: (n,3)
    :param edge_list: [[edge0_v0, edge0_v1], [edge1_v0, edge1_v1], ...]
    :return: the minimum distance and projection point
    """
    min_distance = float('inf')
    min_projetion = np.zeros(3)
    for edge in edge_list:
        edge_start, edge_end = edge[0], edge[1]  # Assuming edge is a tuple/list of two vertices
        distance, projection = distance_point_to_edge(contact_point, edge_start, edge_end)
        if distance < min_distance:
            min_distance = distance
            min_projetion = projection
    return min_distance, min_projetion


def points_obb(pointsarray, toggledebug=False):
    """
    applicable to both 2d and 3d pointsarray
    :param pointsarray: nx3 or nx3 array
    :return: center, corners, and [x, y, ...] frame
    author: weiwei
    date: 20191229, 20200701osaka
    """
    pcv, pcaxmat = compute_pca(pointsarray)
    pcaxmat_t = pcaxmat.T
    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    ar = np.dot(pointsarray, np.linalg.inv(pcaxmat_t))
    # get the minimum and maximum
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    diff = (maxa - mina) * 0.5
    # the center is just half way between the min and max xy
    center = mina + diff
    # get the corners by subtracting and adding half the bounding boxes height and width to the center
    if pointsarray.shape[1] == 2:
        corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]],
                            center + [diff[0], diff[1]], center + [-diff[0], diff[1]]])
    elif pointsarray.shape[1] == 3:
        corners = np.array([center + [-diff[0], -diff[1], -diff[2]], center + [diff[0], -diff[1], -diff[2]],
                            center + [diff[0], diff[1], -diff[2]], center + [-diff[0], diff[1], -diff[2]],
                            center + [-diff[0], diff[1], diff[2]], center + [-diff[0], -diff[1], diff[2]],
                            center + [diff[0], -diff[1], diff[2]], center + [diff[0], diff[1], diff[2]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners, pcaxmat_t)
    center = np.dot(center, pcaxmat_t)
    if toggledebug:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        ax.scatter(pointsarray[:, 0], pointsarray[:, 1])
        ax.scatter([center[0]], [center[1]])
        ax.plot(corners[:, 0], corners[:, 1], '-')
        plt.axis('equal')
        plt.show()
    return [center, corners, pcaxmat]


def gaussian_ellipsoid(pointsarray):
    """
    compute a 95% percent ellipsoid axes_mat for the given points array
    :param pointsarray:
    :return:
    author: weiwei
    date: 20200701
    """
    pcv, pcaxmat = compute_pca(pointsarray)
    center = np.mean(pointsarray, axis=0)
    axmat = np.eye(3)
    # TODO is there a better way to do this?
    axmat[:, 0] = 2 * math.sqrt(5.991 * pcv[0]) * pcaxmat[:, 0]
    axmat[:, 1] = 2 * math.sqrt(5.991 * pcv[1]) * pcaxmat[:, 1]
    axmat[:, 2] = 2 * math.sqrt(5.991 * pcv[2]) * pcaxmat[:, 2]
    return center, axmat


def random_rgba(toggle_alpha_random=False):
    """
    randomize a 1x4 list in range 0-1
    :param toggle_alpha_random: alpha = 1 if False
    :return: 
    """
    if not toggle_alpha_random:
        return np.random.random_sample(3).tolist() + [1]
    else:
        return np.random.random_sample(4).tolist()


def consecutive(nparray1d, stepsize=1):
    """
    find consecutive sequences from an array
    example:
    a = np.array([0, 47, 48, 49, 50, 97, 98, 99])
    consecutive(a)
    returns [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])]
    :param nparray1d:
    :param stepsize:
    :return:
    """
    return np.split(nparray1d, np.where(np.diff(nparray1d) != stepsize)[0] + 1)


def null_space(npmat):
    return scipy.linalg.null_space(npmat)


def homopos(pos: np.ndarray):
    """
    append 1 to pos
    :param pos:
    :return:
    """
    return np.array([pos[0], pos[1], pos[2], 1])


def reflection_homomat(point, normal):
    """
    Return matrix to mirror at plane defined by point and normal vector.
    """
    normal = unit_vector(normal[:3])
    homomat = np.identity(4)
    homomat[:3, :3] -= 2.0 * np.outer(normal, normal)
    homomat[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return homomat


def reflection_from_homomat(homomat):
    """
    Return mirror plane point and normal vector from reflection homomat
    """
    homomat = np.array(homomat, dtype=np.float64, copy=False)
    # normal: unit eigenvector corresponding to eigenvalue -1
    w, v = np.linalg.eig(homomat[:3, :3])
    i = np.where(abs(np.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = np.real(v[:, i[0]]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    w, v = np.linalg.eig(homomat)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(v[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def projection_homomat(point, normal, perspective=None, pseudo=False):
    """
    Return matrix to project onto plane defined by point and normal.
    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).
    """
    homomat = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64,
                               copy=False)
        homomat[0, 0] = homomat[1, 1] = homomat[2, 2] = np.dot(perspective - point, normal)
        homomat[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            homomat[:3, :3] -= np.outer(normal, normal)
            homomat[:3, 3] = np.dot(point, normal) * (perspective + normal)
        else:
            homomat[:3, 3] = np.dot(point, normal) * perspective
        homomat[3, :3] = -normal
        homomat[3, 3] = np.dot(perspective, normal)
    else:
        # orthogonal projection
        homomat[:3, :3] -= np.outer(normal, normal)
        homomat[:3, 3] = np.dot(point, normal) * normal
    return homomat

def affine_matrix_from_points(v0, v1, shear=True, scale=True, use_svd=True):
    """
    Return affine transform matrix to register two point sets.
    v0 and v1 are shape (n_dims, ) arrays of at least n_dims non-homogeneous
    coordinates, where n_dims is the dimensionality of the coordinate space.
    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix is returned.
    By default the algorithm by Hartley and Zissermann [15] is used.
    If use_svd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if n_dims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.
    The returned matrix performs rotation, translation and uniform scaling(if specified).
    v1 = return_value @ v0
    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)
    n_dims = v0.shape[0]
    if n_dims < 2 or v0.shape[1] < n_dims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")
    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(n_dims + 1)
    M0[:n_dims, n_dims] = t0
    v0 += t0.reshape(n_dims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(n_dims + 1)
    M1[:n_dims, n_dims] = t1
    v1 += t1.reshape(n_dims, 1)
    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:n_dims].T
        B = vh[:n_dims]
        C = vh[n_dims:2 * n_dims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((n_dims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * n_dims) + (1.0,)))
    elif use_svd or n_dims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, n_dims - 1], vh[n_dims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(n_dims + 1)
        M[:n_dims, :n_dims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx + yy + zz, 0.0, 0.0, 0.0],
             [yz - zy, xx - yy - zz, 0.0, 0.0],
             [zx - xz, xy + yx, yy - xx - zz, 0.0],
             [xy - yx, zx + xz, yz + zy, zz - xx - yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= np.linalg.norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = homomat_from_quaternion(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:n_dims, :n_dims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[n_dims, n_dims]
    return M


# def _unit_vector(data, axis=None, out=None):
#     """Return ndarray normalized by axis_length, i.e. Euclidean norm, along axis.
#
#     >>> v0 = np.random.random(3)
#     >>> v1 = unit_vector(v0)
#     >>> np.allclose(v1, v0 / np.linalg.norm(v0))
#     True
#     >>> v0 = np.random.rand(5, 4, 3)
#     >>> v1 = unit_vector(v0, axis=-1)
#     >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
#     >>> np.allclose(v1, v2)
#     True
#     >>> v1 = unit_vector(v0, axis=1)
#     >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
#     >>> np.allclose(v1, v2)
#     True
#     >>> v1 = np.empty((5, 4, 3))
#     >>> unit_vector(v0, axis=1, out=v1)
#     >>> np.allclose(v1, v2)
#     True
#     >>> list(unit_vector([]))
#     []
#     >>> list(unit_vector([1]))
#     [1.0]
#
#     """
#     if out is None:
#         data = np.array(data, dtype=np.float64, copy=True)
#         if data.ndim == 1:
#             data /= math.sqrt(np.dot(data, data))
#             return data
#     else:
#         if out is not data:
#             out[:] = np.array(data, copy=False)
#         data = out
#     length = np.atleast_1d(np.sum(data * data, axis))
#     np.sqrt(length, length)
#     if axis is not None:
#         length = np.expand_dims(length, axis)
#     data /= length
#     if out is None:
#         return data


# class Arcball(object):
#     """Virtual Trackball Control.
#
#     >>> ball = Arcball()
#     >>> ball = Arcball(initial=np.identity(4))
#     >>> ball.place([320, 320], 320)
#     >>> ball.down([500, 250])
#     >>> ball.drag([475, 275])
#     >>> R = ball.matrix()
#     >>> np.allclose(np.sum(R), 3.90583455)
#     True
#     >>> ball = Arcball(initial=[1, 0, 0, 0])
#     >>> ball.place([320, 320], 320)
#     >>> ball.setaxes([1, 1, 0], [-1, 1, 0])
#     >>> ball.constrain = True
#     >>> ball.down([400, 200])
#     >>> ball.drag([200, 400])
#     >>> R = ball.matrix()
#     >>> np.allclose(np.sum(R), 0.2055924)
#     True
#     >>> ball.next()
#
#     """
#
#     def __init__(self, initial=None):
#         """Initialize virtual trackball control.
#
#         initial : quaternion or rotation matrix
#
#         """
#         self._axis = None
#         self._axes = None
#         self._radius = 1.0
#         self._center = [0.0, 0.0]
#         self._vdown = np.array([0.0, 0.0, 1.0])
#         self._constrain = False
#         if initial is None:
#             self._qdown = np.array([1.0, 0.0, 0.0, 0.0])
#         else:
#             initial = np.array(initial, dtype=np.float64)
#             if initial.shape == (4, 4):
#                 self._qdown = quaternion_from_matrix(initial)
#             elif initial.shape == (4,):
#                 initial /= vector_norm(initial)
#                 self._qdown = initial
#             else:
#                 raise ValueError("initial not a quaternion or matrix")
#         self._qnow = self._qpre = self._qdown
#
#     def place(self, center, radius):
#         """Place Arcball, e.g. when window size changes.
#
#         center : sequence[2]
#             Window coordinates of trackball center.
#         major_radius : float
#             Radius of trackball in window coordinates.
#
#         """
#         self._radius = float(radius)
#         self._center[0] = center[0]
#         self._center[1] = center[1]
#
#     def setaxes(self, *axes):
#         """Set axes to constrain rotations."""
#         if axes is None:
#             self._axes = None
#         else:
#             self._axes = [_unit_vector(axis) for axis in axes]
#
#     @property
#     def constrain(self):
#         """Return state of constrain to axis mode."""
#         return self._constrain
#
#     @constrain.setter
#     def constrain(self, value):
#         """Set state of constrain to axis mode."""
#         self._constrain = bool(value)
#
#     def down(self, point):
#         """Set initial cursor window coordinates and pick constrain-axis."""
#         self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
#         self._qdown = self._qpre = self._qnow
#         if self._constrain and self._axes is not None:
#             self._axis = arcball_nearest_axis(self._vdown, self._axes)
#             self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
#         else:
#             self._axis = None
#
#     def drag(self, point):
#         """Update current cursor window coordinates."""
#         vnow = arcball_map_to_sphere(point, self._center, self._radius)
#         if self._axis is not None:
#             vnow = arcball_constrain_to_axis(vnow, self._axis)
#         self._qpre = self._qnow
#         t = np.cross(self._vdown, vnow)
#         if np.dot(t, t) < _EPS:
#             self._qnow = self._qdown
#         else:
#             q = [np.dot(self._vdown, vnow), t[0], t[1], t[2]]
#             self._qnow = quaternion_multiply(q, self._qdown)
#
#     def next(self, acceleration=0.0):
#         """Continue rotation in motion_vec of last drag."""
#         q = quaternion_slerp(self._qpre, self._qnow, 2.0 + acceleration, False)
#         self._qpre, self._qnow = self._qnow, q
#
#     def matrix(self):
#         """Return homogeneous rotation matrix."""
#         return quaternion_matrix(self._qnow)


# def arcball_map_to_sphere(point, center, radius):
#     """Return unit sphere coordinates from window coordinates."""
#     v0 = (point[0] - center[0]) / radius
#     v1 = (center[1] - point[1]) / radius
#     n = v0 * v0 + v1 * v1
#     if n > 1.0:
#         # position outside of sphere
#         n = math.sqrt(n)
#         return np.array([v0 / n, v1 / n, 0.0])
#     else:
#         return np.array([v0, v1, math.sqrt(1.0 - n)])
#
#
# def arcball_constrain_to_axis(point, axis):
#     """Return sphere point perpendicular to axis."""
#     v = np.array(point, dtype=np.float64, copy=True)
#     a = np.array(axis, dtype=np.float64, copy=True)
#     v -= a * np.dot(a, v)  # on plane
#     n = vector_norm(v)
#     if n > _EPS:
#         if v[2] < 0.0:
#             np.negative(v, v)
#         v /= n
#         return v
#     if a[2] == 1.0:
#         return np.array([1.0, 0.0, 0.0])
#     return _unit_vector([-a[1], a[0], 0.0])
#
#
# def arcball_nearest_axis(point, axes):
#     """Return axis, which arc is nearest to point."""
#     point = np.array(point, dtype=np.float64, copy=False)
#     nearest = None
#     mx = -1.0
#     for axis in axes:
#         t = np.dot(arcball_constrain_to_axis(point, axis), point)
#         if t > mx:
#             nearest = axis
#             mx = t
#     return nearest


# if __name__ == '__main__':
#     start_pos = np.array([1, 0, 0])
#     start_rotmat = np.eye(3)
#     goal_pos = np.array([2, 0, 0])
#     goal_rotmat = np.eye(3)
#     pos_list, rotmat_list = interplate_pos_rotmat(start_pos, start_rotmat, goal_pos, goal_rotmat, granularity=3)
#     print(pos_list, rotmat_list)
