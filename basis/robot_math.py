import math
import scipy
import operator
import warnings
import functools
import numpy as np
import numpy.typing as npt
from sklearn import cluster
import basis.trimesh.creation as trm_cr
import basis.constant as bc
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0
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


# helper
def radians(degree_val):
    return np.radians(degree_val)


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
    pass


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


def rotmat_from_euler(ai, aj, ak, axes='sxyz'):
    """
    :param ai: radian
    :param aj: radian
    :param ak: radian
    :param axes:
    :return:
    author: weiwei
    date: 20190504
    """
    return _euler_matrix(ai, aj, ak, axes)[:3, :3]


def rotmat_to_euler(rotmat, axes='sxyz'):
    """
    :param rotmat: 3x3 nparray
    :param axes: order
    :return: radian
    author: weiwei
    date: 20190504
    """
    ax, ay, az = _euler_from_matrix(rotmat, axes)
    return np.array([ax, ay, az])


def rotmat_between_vectors(v1, v2):
    """
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
        quaternion_list.append(quaternion_from_matrix(rotmat))
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
    homomat = np.eye(4, 4)
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
    invhomomat = np.eye(4)
    invhomomat[:3, :3] = np.transpose(rotmat)
    invhomomat[:3, 3] = -np.dot(np.transpose(rotmat), pos)
    return invhomomat


def homomat_average(homomatlist, bandwidth=10):
    """
    average a list of pos (4x4)
    :param homomatlist:
    :param bandwidth:
    :param denoise:
    :return:
    author: weiwei
    date: 20200109
    """
    homomatarray = np.asarray(homomatlist)
    posavg = pos_average(homomatarray[:, :3, 3], bandwidth)
    rotmatavg = rotmat_average(homomatarray[:, :3, :3], bandwidth)
    return homomat_from_posrot(posavg, rotmatavg)


def homomat_from_quaternion(quaternion):
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
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle / 2.0) / qlen
    quaternion[0] = math.cos(angle / 2.0)
    return quaternion


def quaternion_average(quaternionlist, bandwidth=10):
    """
    average a list of quaternion (nx4)
    this is the full version
    :param rotmatlist:
    :param bandwidth: meanshift denoising is applied if available
    :return:
    author: weiwei
    date: 20190422
    """
    if len(quaternionlist) == 0:
        return False
    quaternionarray = np.array(quaternionlist)
    if bandwidth is not None:
        anglelist = []
        for quaternion in quaternionlist:
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


def quaternion_to_euler(quaternion, axes='sxyz'):
    """
    :param rotmat: 3x3 nparray
    :param axes: order
    :return: radian
    author: weiwei
    date: 20190504
    """
    return rotmat_to_euler(rotmat_from_quaternion(quaternion), axes)


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
        return None
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


def delta_w_between_rotmat(src_rotmat, tgt_rotmat):
    """
    compute angle*ax from src_rotmat to tgt_rotmat
    the following relation holds for the returned delta_w
    rotmat_from_axangle(np.linalg.norm(deltaw), unit_vec(deltaw)).dot(src_rotmat) = tgt_rotmat
    :param src_rotmat: 3x3 nparray
    :param tgt_rotmat: 3x3 nparray
    :return:
    author: weiwei
    date: 20200326
    """
    delta_rotmat = tgt_rotmat @ src_rotmat.T
    tmp_vec = np.array([delta_rotmat[2, 1] - delta_rotmat[1, 2],
                        delta_rotmat[0, 2] - delta_rotmat[2, 0],
                        delta_rotmat[1, 0] - delta_rotmat[0, 1]])
    tmp_vec_norm = np.linalg.norm(tmp_vec)
    if tmp_vec_norm > 1e-6:
        delta_w = math.atan2(tmp_vec_norm, np.trace(delta_rotmat) - 1.0) / tmp_vec_norm * tmp_vec
    elif delta_rotmat[0, 0] > 0 and delta_rotmat[1, 1] > 0 and delta_rotmat[2, 2] > 0:
        delta_w = np.array([0, 0, 0])
    else:
        delta_w = np.pi / 2 * (np.diag(delta_rotmat) + 1)
    return delta_w


def diff_between_posrot(src_pos,
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
    icos = trm_cr.icosphere(icolevel)
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
    rot_angles = np.linspace(0, 2 * np.pi, np.pi * 2 / rotation_interval, endpoint=False)
    returnlist = []
    icos = trm_cr.icosphere(icolevel)
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


def get_rgba_from_cmap(id, cm_name='tab20', step=20):
    """
    get rgba from matplotlib cmap "tab20"
    :param id:
    :param cm_name: see matplotlib tutorials
    :param step:
    :return:
    author: weiwei
    date: 20210505, 20220404
    """
    cm = plt.get_cmap(cm_name)
    return list(cm(id % step))


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


# The following code is from Gohlke
#
#
# Copyright (c) 2006-2015, Christoph Gohlke
# Copyright (c) 2006-2015, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Homogeneous Transformation Matrices and Quaternions.

A library for calculating 4x4 matrices for translating, rotating, reflecting,
scaling, shearing, projecting, orthogonalizing, and superimposing arrays of
3D homogeneous coordinates as well as for converting between rotation matrices,
Euler angles, and quaternions. Also includes an Arcball control object and
functions to decompose transformation matrices.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2015.03.19

Requirements
------------
* `CPython 2.7 or 3.4 <http://www.python.org>`_
* `Numpy 1.9 <http://www.np.org>`_
* `Transformations.c 2015.03.19 <http://www.lfd.uci.edu/~gohlke/>`_
  (recommended for speedup of some functions)

Notes
-----
The API is not stable yet and is expected to change between revisions.

This Python code is not optimized for speed. Refer to the transformations.c
module for a faster implementation of some functions.

Documentation in HTML format can be generated with epydoc.

Matrices (M) can be inverted using np.linalg.inv(M), be concatenated using
np.dot(M0, M1), or transform homogeneous coordinate arrays (v) using
np.dot(M, v) for shape (4, \*) column vectors, respectively
np.dot(v, M.T) for shape (\*, 4) row vectors ("array of points").

This module follows the "column vectors on the right" and "row major storage"
(C contiguous) conventions. The translation components are in the right column
of the transformation matrix, i.e. M[:3, 3].
The transpose of the transformation matrices may have to be used to interface
with other graphics systems, e.g. with OpenGL's glMultMatrixd(). See also [16].

Calculations are carried out with np.float64 precision.

Vector, point, quaternion, and matrix function arguments are expected to be
"array like", i.e. tuple, list, or numpy arrays.

Return types are numpy arrays unless specified otherwise.

Angles are in radians unless specified otherwise.

Quaternions w+ix+jy+kz are represented as [w, x, y, z].

A triple of Euler angles can be applied/interpreted in 24 ways, which can
be specified using a 4 character string or encoded 4-tuple:

  *Axes 4-string*: e.g. 'sxyz' or 'ryxy'

  - first character : rotations are applied to 's'tatic or 'r'otating frame
  - remaining characters : successive rotation axis 'x', 'y', or 'z'

  *Axes 4-tuple*: e.g. (0, 0, 0, 0) or (1, 1, 1, 1)

  - inner axis: code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
  - parity : even (0) if inner axis 'x' is followed by 'y', 'y' is followed
    by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
  - repetition : first and last axis are same (1) or different (0).
  - frame : rotations are applied to static (0) or rotating (1) frame.

Other Python packages and modules for 3D transformations and quaternions:

* `Transforms3d <https://pypi.python.org/pypi/transforms3d>`_
   includes most code of this module.
* `Blender.mathutils <http://www.blender.org/api/blender_python_api>`_
* `numpy-dtypes <https://github.com/numpy/numpy-dtypes>`_

References
----------
(1)  Matrices and transformations. Ronald Goldman.
     In "Graphics Gems I", pp 472-475. Morgan Kaufmann, 1990.
(2)  More matrices and transformations: shear and pseudo-perspective.
     Ronald Goldman. In "Graphics Gems II", pp 320-323. Morgan Kaufmann, 1991.
(3)  Decomposing a matrix into simple transformations. Spencer Thomas.
     In "Graphics Gems II", pp 320-323. Morgan Kaufmann, 1991.
(4)  Recovering the data from the transformation matrix. Ronald Goldman.
     In "Graphics Gems II", pp 324-331. Morgan Kaufmann, 1991.
(5)  Euler angle conversion. Ken Shoemake.
     In "Graphics Gems IV", pp 222-229. Morgan Kaufmann, 1994.
(6)  Arcball rotation control. Ken Shoemake.
     In "Graphics Gems IV", pp 175-192. Morgan Kaufmann, 1994.
(7)  Representing attitude: Euler angles, unit quaternions, and rotation
     vectors. James Diebel. 2006.
(8)  A discussion of the solution for the best rotation to relate two sets
     of vectors. W Kabsch. Acta Cryst. 1978. A34, 827-828.
(9)  Closed-form solution of absolute orientation using unit quaternions.
     BKP Horn. J Opt Soc Am A. 1987. 4(4):629-642.
(10) Quaternions. Ken Shoemake.
     http://www.sfu.ca/~jwa3/cmpt461/files/quatut.pdf
(11) From quaternion to matrix and back. JMP van Waveren. 2005.
     http://www.intel.com/cd/ids/developer/asmo-na/eng/293748.htm
(12) Uniform random rotations. Ken Shoemake.
     In "Graphics Gems III", pp 124-132. Morgan Kaufmann, 1992.
(13) Quaternion in molecular modeling. CFF Karney.
     J Mol Graph Mod, 25(5):595-604
(14) New method for extracting the quaternion from a rotation matrix.
     Itzhack Y Bar-Itzhack, J Guid Contr Dynam. 2000. 23(6): 1085-1087.
(15) Multiple View Geometry in Computer Vision. Hartley and Zissermann.
     Cambridge University Press; 2nd Ed. 2004. Chapter 4, Algorithm 4.7, p 130.
(16) Column Vectors vs. Row Vectors.
     http://steve.hollasch.net/cgindex/math/matrix/column-vec.html

Examples
--------
>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>> origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
>>> I = identity_matrix()
>>> Rx = rotation_matrix(alpha, xaxis)
>>> Ry = rotation_matrix(beta, yaxis)
>>> Rz = rotation_matrix(gamma, zaxis)
>>> R = concatenate_matrices(Rx, Ry, Rz)
>>> euler = _euler_from_matrix(R, 'rxyz')
>>> np.allclose([alpha, beta, gamma], euler)
True
>>> Re = _euler_matrix(alpha, beta, gamma, 'rxyz')
>>> is_same_transform(R, Re)
True
>>> al, be, ga = _euler_from_matrix(Re, 'rxyz')
>>> is_same_transform(Re, _euler_matrix(al, be, ga, 'rxyz'))
True
>>> qx = quaternion_about_axis(alpha, xaxis)
>>> qy = quaternion_about_axis(beta, yaxis)
>>> qz = quaternion_about_axis(gamma, zaxis)
>>> q = quaternion_multiply(qx, qy)
>>> q = quaternion_multiply(q, qz)
>>> Rq = quaternion_matrix(q)
>>> is_same_transform(R, Rq)
True
>>> S = scale_matrix(1.23, origin)
>>> T = translation_matrix([1, 2, 3])
>>> Z = shear_matrix(beta, xaxis, origin, zaxis)
>>> R = random_rotation_matrix(np.random.rand(3))
>>> M = concatenate_matrices(T, R, Z, S)
>>> scale, shear, angles, trans, persp = decompose_matrix(M)
>>> np.allclose(scale, 1.23)
True
>>> np.allclose(trans, [1, 2, 3])
True
>>> np.allclose(shear, [0, math.tan(beta), 0])
True
>>> is_same_transform(R, _euler_matrix(axes='sxyz', *angles))
True
>>> M1 = compose_matrix(scale, shear, angles, trans, persp)
>>> is_same_transform(M, M1)
True
>>> v0, v1 = random_vector(3), random_vector(3)
>>> M = rotation_matrix(angle_between_vectors(v0, v1), vector_product(v0, v1))
>>> v2 = np.dot(v0, M[:3,:3].T)
>>> np.allclose(unit_vector(v1), unit_vector(v2))
True

"""


def reflection_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector.

    >>> v0 = np.random.random(4) - 0.5
    >>> v0[3] = 1.
    >>> v1 = np.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> np.allclose(2, np.trace(R))
    True
    >>> np.allclose(v0, np.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> np.allclose(v2, np.dot(R, v3))
    True

    """
    normal = _unit_vector(normal[:3])
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return M


def reflection_from_matrix(matrix):
    """Return mirror plane point and normal vector from reflection matrix.

    >>> v0 = np.random.random(3) - 0.5
    >>> v1 = np.random.random(3) - 0.5
    >>> M0 = reflection_matrix(v0, v1)
    >>> point, normal = reflection_from_matrix(M0)
    >>> M1 = reflection_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    # normal: unit eigenvector corresponding to eigenvalue -1
    w, V = np.linalg.eig(M[:3, :3])
    i = np.where(abs(np.real(w) + 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue -1")
    normal = np.real(V[:, i[0]]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return point, normal


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and motion_vec.

    >>> R = rotation_matrix(np.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*np.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*np.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(np.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(np.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = _unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> angle = (random.random() - 0.5) * (2*np.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> angle, direc, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direc, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # motion_vec: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    # rotation angle depending on motion_vec
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, pint


def scale_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in motion_vec.

    Use factor -1 for point symmetry.

    >>> v = (np.random.rand(4, 5) - 0.5) * 20
    >>> v[3] = 1
    >>> S = scale_matrix(-1.234)
    >>> np.allclose(np.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)

    """
    if direction is None:
        # uniform scaling
        M = np.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = _unit_vector(direction[:3])
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M


def scale_from_matrix(matrix):
    """Return scaling factor, origin and motion_vec from scaling matrix.

    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S0 = scale_matrix(factor, origin)
    >>> factor, origin, motion_vec = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, motion_vec)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scale_matrix(factor, origin, direct)
    >>> factor, origin, motion_vec = scale_from_matrix(S0)
    >>> S1 = scale_matrix(factor, origin, motion_vec)
    >>> is_same_transform(S0, S1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    factor = np.trace(M33) - 2.0
    try:
        # motion_vec: unit eigenvector corresponding to eigenvalue factor
        w, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(w) - factor) < 1e-8)[0][0]
        direction = np.real(V[:, i]).squeeze()
        direction /= vector_norm(direction)
    except IndexError:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        direction = None
    # origin: any eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    origin = np.real(V[:, i[-1]]).squeeze()
    origin /= origin[3]
    return factor, origin, direction


def projection_matrix(point, normal, direction=None,
                      perspective=None, pseudo=False):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection motion_vec, or none of both.

    If pseudo is True, perspective projections will preserve relative depth
    such that Perspective = dot(Orthogonal, PseudoPerspective).

    >>> P = projection_matrix([0, 0, 0], [1, 0, 0])
    >>> np.allclose(P[1:, 1:], np.identity(4)[1:, 1:])
    True
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> P1 = projection_matrix(point, normal, motion_vec=direct)
    >>> P2 = projection_matrix(point, normal, perspective=persp)
    >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> is_same_transform(P2, np.dot(P0, P3))
    True
    >>> P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    >>> v0 = (np.random.rand(4, 5) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(P, v0)
    >>> np.allclose(v1[1], v0[1])
    True
    >>> np.allclose(v1[0], 3-v1[1])
    True

    """
    M = np.identity(4)
    point = np.array(point[:3], dtype=np.float64, copy=False)
    normal = _unit_vector(normal[:3])
    if perspective is not None:
        # perspective projection
        perspective = np.array(perspective[:3], dtype=np.float64,
                               copy=False)
        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective - point, normal)
        M[:3, :3] -= np.outer(perspective, normal)
        if pseudo:
            # preserve relative depth
            M[:3, :3] -= np.outer(normal, normal)
            M[:3, 3] = np.dot(point, normal) * (perspective + normal)
        else:
            M[:3, 3] = np.dot(point, normal) * perspective
        M[3, :3] = -normal
        M[3, 3] = np.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = np.array(direction[:3], dtype=np.float64, copy=False)
        scale = np.dot(direction, normal)
        M[:3, :3] -= np.outer(direction, normal) / scale
        M[:3, 3] = direction * (np.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= np.outer(normal, normal)
        M[:3, 3] = np.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix, pseudo=False):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, motion_vec, perspective, and pseudo.

    >>> point = np.random.random(3) - 0.5
    >>> normal = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> persp = np.random.random(3) - 0.5
    >>> P0 = projection_matrix(point, normal)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, direct)
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    >>> result = projection_from_matrix(P0, pseudo=False)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    >>> result = projection_from_matrix(P0, pseudo=True)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not pseudo and len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        # motion_vec: unit eigenvector corresponding to eigenvalue 0
        w, V = np.linalg.eig(M33)
        i = np.where(abs(np.real(w)) < 1e-8)[0]
        if not len(i):
            raise ValueError("no eigenvector corresponding to eigenvalue 0")
        direction = np.real(V[:, i[0]]).squeeze()
        direction /= vector_norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        w, V = np.linalg.eig(M33.T)
        i = np.where(abs(np.real(w)) < 1e-8)[0]
        if len(i):
            # parallel projection
            normal = np.real(V[:, i[0]]).squeeze()
            normal /= vector_norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals motion_vec vector
            return point, direction, None, None, False
    else:
        # perspective projection
        i = np.where(abs(np.real(w)) > 1e-8)[0]
        if not len(i):
            raise ValueError(
                "no eigenvector not corresponding to eigenvalue 0")
        point = np.real(V[:, i[-1]]).squeeze()
        point /= point[3]
        normal = - M[3, :3]
        perspective = M[:3, 3] / np.dot(point[:3], normal)
        if pseudo:
            perspective -= normal
        return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
    """Return matrix to obtain normalized device coordinates from frustum.

    The frustum bounds are axis-aligned along x (left, right),
    y (bottom, top) and z (near, far).

    Normalized device coordinates are in range [-1, 1] if coordinates are
    inside the frustum.

    If perspective is True the frustum is a truncated pyramid with the
    perspective point at origin and motion_vec along z axis, otherwise an
    orthographic canonical view volume (a box).

    Homogeneous coordinates transformed by the perspective clip matrix
    need to be dehomogenized (divided by w coordinate).

    >>> frustum = np.random.rand(6)
    >>> frustum[1] += frustum[0]
    >>> frustum[3] += frustum[2]
    >>> frustum[5] += frustum[4]
    >>> M = clip_matrix(perspective=False, *frustum)
    >>> np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    array([-1., -1., -1.,  1.])
    >>> np.dot(M, [frustum[1], frustum[3], frustum[5], 1])
    array([ 1.,  1.,  1.,  1.])
    >>> M = clip_matrix(perspective=True, *frustum)
    >>> v = np.dot(M, [frustum[0], frustum[2], frustum[4], 1])
    >>> v / v[3]
    array([-1., -1., -1.,  1.])
    >>> v = np.dot(M, [frustum[1], frustum[3], frustum[4], 1])
    >>> v / v[3]
    array([ 1.,  1., -1.,  1.])

    """
    if left >= right or bottom >= top or near >= far:
        raise ValueError("invalid frustum")
    if perspective:
        if near <= _EPS:
            raise ValueError("invalid frustum: near <= 0")
        t = 2.0 * near
        M = [[t / (left - right), 0.0, (right + left) / (right - left), 0.0],
             [0.0, t / (bottom - top), (top + bottom) / (top - bottom), 0.0],
             [0.0, 0.0, (far + near) / (near - far), t * far / (far - near)],
             [0.0, 0.0, -1.0, 0.0]]
    else:
        M = [[2.0 / (right - left), 0.0, 0.0, (right + left) / (left - right)],
             [0.0, 2.0 / (top - bottom), 0.0, (top + bottom) / (bottom - top)],
             [0.0, 0.0, 2.0 / (far - near), (far + near) / (near - far)],
             [0.0, 0.0, 0.0, 1.0]]
    return np.array(M)


def shear_matrix(angle, direction, point, normal):
    """Return matrix to shear by angle along motion_vec vector on shear plane.

    The shear plane is defined by a point and normal vector. The motion_vec
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the motion_vec vector and its xyz_lengths is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> angle = (random.random() - 0.5) * 4*np.pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S = shear_matrix(angle, direct, point, normal)
    >>> np.allclose(1, np.linalg.det(S))
    True

    """
    normal = _unit_vector(normal[:3])
    direction = _unit_vector(direction[:3])
    if abs(np.dot(normal, direction)) > 1e-6:
        raise ValueError("motion_vec and normal vectors are not orthogonal")
    angle = math.tan(angle)
    M = np.identity(4)
    M[:3, :3] += angle * np.outer(direction, normal)
    M[:3, 3] = -angle * np.dot(point[:3], normal) * direction
    return M


def shear_from_matrix(matrix):
    """Return shear angle, motion_vec and plane from shear matrix.

    >>> angle = (random.random() - 0.5) * 4*np.pi
    >>> direct = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> normal = np.cross(direct, np.random.random(3))
    >>> S0 = shear_matrix(angle, direct, point, normal)
    >>> angle, direct, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direct, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    M33 = M[:3, :3]
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    w, V = np.linalg.eig(M33)
    i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0]
    if len(i) < 2:
        raise ValueError("no two linear independent eigenvectors found %s" % w)
    V = np.real(V[:, i]).squeeze().T
    lenorm = -1.0
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        n = np.cross(V[i0], V[i1])
        w = vector_norm(n)
        if w > lenorm:
            lenorm = w
            normal = n
    normal /= lenorm
    # motion_vec and angle
    direction = np.dot(M33 - np.identity(3), normal)
    angle = vector_norm(direction)
    direction /= angle
    angle = math.atan(angle)
    # point: eigenvector corresponding to eigenvalue 1
    w, V = np.linalg.eig(M)
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no eigenvector corresponding to eigenvalue 1")
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    Raise ValueError if matrix is of wrong end_type or degenerative.

    >>> T0 = translation_matrix([1, 2, 3])
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> np.allclose(T0, T1)
    True
    >>> S = scale_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = _euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = _euler_matrix(*angles)
    >>> np.allclose(R0, R1)
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0.0, 0.0, 0.0, 1.0
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3,))
    shear = [0.0, 0.0, 0.0]
    angles = [0.0, 0.0, 0.0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0.0, 0.0, 0.0, 1.0
    else:
        perspective = np.array([0.0, 0.0, 0.0, 1.0])

    translate = M[3, :3].copy()
    M[3, :3] = 0.0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        np.negative(scale, scale)
        np.negative(row, row)

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """Return transformation matrix from sequence of transformations.

    This is the inverse of the decompose_matrix function.

    Sequence of transformations:
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    >>> scale = np.random.random(3) - 0.5
    >>> shear = np.random.random(3) - 0.5
    >>> angles = (np.random.random(3) - 0.5) * (2*np.pi)
    >>> trans = np.random.random(3) - 0.5
    >>> persp = np.random.random(4) - 0.5
    >>> M0 = compose_matrix(scale, shear, angles, trans, persp)
    >>> result = decompose_matrix(M0)
    >>> M1 = compose_matrix(*result)
    >>> is_same_transform(M0, M1)
    True

    """
    M = np.identity(4)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]
        M = np.dot(M, P)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]
        M = np.dot(M, T)
    if angles is not None:
        R = _euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = np.dot(M, R)
    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)
    M /= M[3, 3]
    return M


def orthogonalization_matrix(lengths, angles):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    Angles are expected in radian.

    The de-orthogonalization matrix is the inverse.

    >>> O = orthogonalization_matrix([10, 10, 10], [90, 90, 90])
    >>> np.allclose(O[:3, :3], np.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> np.allclose(np.sum(O), 43.063229)
    True

    """
    a, b, c = lengths
    angles = np.radians(angles)
    sina, sinb, _ = np.sin(angles)
    cosa, cosb, cosg = np.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return np.array([
        [a * sinb * math.sqrt(1.0 - co * co), 0.0, 0.0, 0.0],
        [-a * sinb * co, b * sina, 0.0, 0.0],
        [a * cosb, b * cosa, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> np.allclose(v1, np.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or end_type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
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
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    """Return matrix to transform given 3D point set into second point set.

    v0 and v1 are shape (3, \*) or (4, \*) arrays of at least 3 points.

    The parameters scale and usesvd are explained in the more general
    affine_matrix_from_points function.

    The returned matrix is a similarity or Euclidean transformation matrix.
    This function has a fast C implementation in transformations.c.

    >>> v0 = np.random.rand(3, 10)
    >>> M = superimposition_matrix(v0, v0)
    >>> np.allclose(M, np.identity(4))
    True
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> v0 = [[1,0,0], [0,1,0], [0,0,1], [1,1,1]]
    >>> v1 = np.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(R, v0)
    >>> M = superimposition_matrix(v0, v1)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> S = scale_matrix(random.random())
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> M = concatenate_matrices(T, R, S)
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-9, 300).reshape(3, -1)
    >>> M = superimposition_matrix(v0, v1, scale=True)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> np.allclose(v1, np.dot(M, v0))
    True
    >>> v = np.empty((4, 100, 3))
    >>> v[:, :, 0] = v0
    >>> M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    >>> np.allclose(v1, np.dot(M, v[:, :, 0]))
    True

    """
    v0 = np.array(v0, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1, dtype=np.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False,
                                     scale=scale, usesvd=usesvd)


def _euler_matrix(ai, aj, ak, axes='sxyz'):
    """
    Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    examples:
        R = _euler_matrix(1, 2, 3, 'syxz')
        np.allclose(np.sum(R[0]), -1.34786452) > True
        R = _euler_matrix(1, 2, 3, (0, 1, 0, 1))
        np.allclose(np.sum(R[0]), -0.383436184) > True
        ai, aj, ak = (4*np.pi) * (np.random.random(3) - 0.5)
        for axes in _AXES2TUPLE.keys():
            R = _euler_matrix(ai, aj, ak, axes)
        for axes in _TUPLE2AXES.keys():
            R = _euler_matrix(ai, aj, ak, axes)

    author: weiwei dapted from Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>
    date: 20200704
    """

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

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

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def _euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    Example:
    R0 = _euler_matrix(1, 2, 3, 'syxz')
    al, be, ga = _euler_from_matrix(R0, 'syxz')
    R1 = _euler_matrix(al, be, ga, 'syxz')
    np.allclose(R0, R1) -> True
    angles = (4*np.pi) * (np.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
        R0 = _euler_matrix(axes=axes, *angles)
        R1 = _euler_matrix(axes=axes, *_euler_from_matrix(R0, axes))
        if not np.allclose(R0, R1): print(axes, "failed")
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

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
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, [1, 0, 0])
    >>> np.allclose(q, [0.99810947, 0.06146124, 0, 0])
    True

    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = vector_norm(q)
    if qlen > _EPS:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)
    return q


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> np.allclose(q, [28, -44, -14, 48])
    True

    """
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q


def quaternion_inverse(quaternion):
    """Return inverse of quaternion.

    >>> q0 = random_quaternion()
    >>> q1 = quaternion_inverse(q0)
    >>> np.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)


def quaternion_real(quaternion):
    """Return real part of quaternion.

    >>> quaternion_real([3, 0, 1, 2])
    3.0

    """
    return float(quaternion[0])


def quaternion_imag(quaternion):
    """Return imaginary part of quaternion.

    >>> quaternion_imag([3, 0, 1, 2])
    array([ 0.,  1.,  2.])

    """
    return np.array(quaternion[1:4], dtype=np.float64, copy=True)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.

    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, math.acos(-np.dot(q0, q1)) / angle)
    True

    """
    q0 = _unit_vector(quat0[:4])
    q1 = _unit_vector(quat1[:4])
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


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.

    >>> q = random_quaternion()
    >>> np.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(np.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)

    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array([np.cos(t2) * r2, np.sin(t1) * r1,
                     np.cos(t1) * r1, np.sin(t2) * r2])


def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.

    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True

    """
    return quaternion_matrix(random_quaternion(rand))


class Arcball(object):
    """Virtual Trackball Control.

    >>> ball = Arcball()
    >>> ball = Arcball(initial=np.identity(4))
    >>> ball.place([320, 320], 320)
    >>> ball.down([500, 250])
    >>> ball.drag([475, 275])
    >>> R = ball.matrix()
    >>> np.allclose(np.sum(R), 3.90583455)
    True
    >>> ball = Arcball(initial=[1, 0, 0, 0])
    >>> ball.place([320, 320], 320)
    >>> ball.setaxes([1, 1, 0], [-1, 1, 0])
    >>> ball.constrain = True
    >>> ball.down([400, 200])
    >>> ball.drag([200, 400])
    >>> R = ball.matrix()
    >>> np.allclose(np.sum(R), 0.2055924)
    True
    >>> ball.next()

    """

    def __init__(self, initial=None):
        """Initialize virtual trackball control.

        initial : quaternion or rotation matrix

        """
        self._axis = None
        self._axes = None
        self._radius = 1.0
        self._center = [0.0, 0.0]
        self._vdown = np.array([0.0, 0.0, 1.0])
        self._constrain = False
        if initial is None:
            self._qdown = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            initial = np.array(initial, dtype=np.float64)
            if initial.shape == (4, 4):
                self._qdown = quaternion_from_matrix(initial)
            elif initial.shape == (4,):
                initial /= vector_norm(initial)
                self._qdown = initial
            else:
                raise ValueError("initial not a quaternion or matrix")
        self._qnow = self._qpre = self._qdown

    def place(self, center, radius):
        """Place Arcball, e.g. when window size changes.

        center : sequence[2]
            Window coordinates of trackball center.
        major_radius : float
            Radius of trackball in window coordinates.

        """
        self._radius = float(radius)
        self._center[0] = center[0]
        self._center[1] = center[1]

    def setaxes(self, *axes):
        """Set axes to constrain rotations."""
        if axes is None:
            self._axes = None
        else:
            self._axes = [_unit_vector(axis) for axis in axes]

    @property
    def constrain(self):
        """Return state of constrain to axis mode."""
        return self._constrain

    @constrain.setter
    def constrain(self, value):
        """Set state of constrain to axis mode."""
        self._constrain = bool(value)

    def down(self, point):
        """Set initial cursor window coordinates and pick constrain-axis."""
        self._vdown = arcball_map_to_sphere(point, self._center, self._radius)
        self._qdown = self._qpre = self._qnow
        if self._constrain and self._axes is not None:
            self._axis = arcball_nearest_axis(self._vdown, self._axes)
            self._vdown = arcball_constrain_to_axis(self._vdown, self._axis)
        else:
            self._axis = None

    def drag(self, point):
        """Update current cursor window coordinates."""
        vnow = arcball_map_to_sphere(point, self._center, self._radius)
        if self._axis is not None:
            vnow = arcball_constrain_to_axis(vnow, self._axis)
        self._qpre = self._qnow
        t = np.cross(self._vdown, vnow)
        if np.dot(t, t) < _EPS:
            self._qnow = self._qdown
        else:
            q = [np.dot(self._vdown, vnow), t[0], t[1], t[2]]
            self._qnow = quaternion_multiply(q, self._qdown)

    def next(self, acceleration=0.0):
        """Continue rotation in motion_vec of last drag."""
        q = quaternion_slerp(self._qpre, self._qnow, 2.0 + acceleration, False)
        self._qpre, self._qnow = self._qnow, q

    def matrix(self):
        """Return homogeneous rotation matrix."""
        return quaternion_matrix(self._qnow)


def arcball_map_to_sphere(point, center, radius):
    """Return unit sphere coordinates from window coordinates."""
    v0 = (point[0] - center[0]) / radius
    v1 = (center[1] - point[1]) / radius
    n = v0 * v0 + v1 * v1
    if n > 1.0:
        # position outside of sphere
        n = math.sqrt(n)
        return np.array([v0 / n, v1 / n, 0.0])
    else:
        return np.array([v0, v1, math.sqrt(1.0 - n)])


def arcball_constrain_to_axis(point, axis):
    """Return sphere point perpendicular to axis."""
    v = np.array(point, dtype=np.float64, copy=True)
    a = np.array(axis, dtype=np.float64, copy=True)
    v -= a * np.dot(a, v)  # on plane
    n = vector_norm(v)
    if n > _EPS:
        if v[2] < 0.0:
            np.negative(v, v)
        v /= n
        return v
    if a[2] == 1.0:
        return np.array([1.0, 0.0, 0.0])
    return _unit_vector([-a[1], a[0], 0.0])


def arcball_nearest_axis(point, axes):
    """Return axis, which arc is nearest to point."""
    point = np.array(point, dtype=np.float64, copy=False)
    nearest = None
    mx = -1.0
    for axis in axes:
        t = np.dot(arcball_constrain_to_axis(point, axis), point)
        if t > mx:
            nearest = axis
            mx = t
    return nearest


def vector_norm(data, axis=None, out=None):
    """Return axis_length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = np.random.random(3)
    >>> n = vector_norm(v)
    >>> np.allclose(n, np.linalg.norm(v))
    True
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def _unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by axis_length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


if __name__ == '__main__':
    start_pos = np.array([1, 0, 0])
    start_rotmat = np.eye(3)
    goal_pos = np.array([2, 0, 0])
    goal_rotmat = np.eye(3)
    pos_list, rotmat_list = interplate_pos_rotmat(start_pos, start_rotmat, goal_pos, goal_rotmat, granularity=3)
    print(pos_list, rotmat_list)
