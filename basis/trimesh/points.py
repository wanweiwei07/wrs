'''
Functions dealing with (n,d) points
'''
import numpy as np
from .constants import log, tol
from .geometry import plane_transform


def transform_points(points, matrix, translate=True):
    """Returns points, rotated by transformation matrix 
    If points is (n,2), matrix must be (3,3)
    if points is (n,3), matrix must be (4,4)
    :param points: nx2 or nx3 set of points
    :param matrix: 3x3 or 4x4
    :param translate: apply translation from matrix or not
    :return:
    author: revised by weiwei
    date: 20201202
    """
    points = np.asanyarray(points, dtype=np.float64)
    if len(points) == 0 or matrix is None:
        return points.copy()
    # check the matrix against the points
    matrix = np.asanyarray(matrix, dtype=np.float64)
    # shorthand the shape
    count, dim = points.shape
    # quickly check to see if we've been passed an identity matrix
    if np.abs(matrix - np.eye(4)[:dim+1, :dim+1]).max() < 1e-8:
        return np.ascontiguousarray(points.copy())
    if translate:
        # apply translation and rotation
        stack = np.column_stack((points, np.ones(count)))
        return np.dot(matrix, stack.T).T[:, :dim]
    # only apply the rotation
    return np.dot(matrix[:dim, :dim], points.T).T


def point_plane_distance(points, plane_normal, plane_origin=[0, 0, 0]):
    w = np.array(points) - plane_origin
    distances = np.dot(plane_normal, w.T) / np.linalg.norm(plane_normal)
    return distances


def major_axis(points):
    """
    Returns an approximate vector representing the major axis of points
    :param points: nxd points
    :return: 1xd vector
    author: revised by weiwei
    date: 20201202
    """
    u, s, v = np.linalg.svd(points)
    axis_guess = v[np.argmax(s)]
    return axis_guess


def surface_normal(points):
    """
    Returns a normal estimate of a group of points using SVD
    :param points: nxd set of points
    :return: 1xd vector
    author: revised by weiwei
    date: 20201202
    """
    normal = np.linalg.svd(points)[2][-1]
    return normal


def plane_fit_lmse(points, tolerance=None):
    """
    TODO: RANSAC version?
    Given a set of points, find an origin and normal using least squares
    :param points: nx3 nparray
    :param tolerance: how non-planar the result can be without raising an error
    :return: C: (3) point on the plane, N: (3) normal vector
    author: revised by weiwei
    date: 20201202
    """
    C = points[0]
    x = points - C
    M = np.dot(x.T, x)
    N = np.linalg.svd(M)[0][:, -1]
    if not (tolerance is None):
        normal_range = np.ptp(np.dot(N, points.T))
        if normal_range > tol.planar:
            log.error('Points have peak to peak of %f', normal_range)
            raise ValueError('Plane outside tolerance!')
    return C, N


def radial_sort(points, origin=None, normal=None):
    """
    Sorts a set of points radially (by angle) around an origin/normal.
    If origin/normal aren't specified, it sorts around centroid
    and the approximate plane the points lie in.
    :param points: nx3
    :param origin:
    :param normal:
    :return:
    author: revised by weiwei
    date: 20201202
    """
    # if origin and normal aren't specified, generate one at the centroid
    if origin == None: origin = np.average(points, axis=0)
    if normal == None: normal = surface_normal(points)
    # create two axis perpendicular to each other and the normal, 
    # and project the points onto them
    axis0 = [normal[0], normal[2], -normal[1]]
    axis1 = np.cross(normal, axis0)
    ptVec = points - origin
    pr0 = np.dot(ptVec, axis0)
    pr1 = np.dot(ptVec, axis1)
    # calculate the angles of the points on the axis
    angles = np.arctan2(pr0, pr1)
    # return the points sorted by angle
    return points[[np.argsort(angles)]]


def project_to_plane(points, plane_normal=[0, 0, 1], plane_origin=[0, 0, 0], transform=None, return_transform=False,
                     return_planar=True):
    """
    Projects a set of nx3 points onto a plane.
    :param points: nx3 nparray
    :param plane_normal: 1x3 nparray
    :param plane_origin: 1x3 nparray
    :param transform: None or 4x4 nparray. If specified, normal/origin are ignored
    :param return_transform: bool, if true returns the 4x4 matrix used to project points onto a plane
    :param return_planar: bool, if True, returns nx2 points. If False, returns nx3, where the Z column consists of zeros
    :return: 
    """
    if np.all(np.abs(plane_normal) < tol.zero):
        raise NameError('Normal must be nonzero!')
    if transform is None:
        transform = plane_transform(plane_origin, plane_normal)
    transformed = transform_points(points, transform)
    transformed = transformed[:, 0:(3 - int(return_planar))]
    if return_transform:
        polygon_to_3D = np.linalg.inv(transform)
        return transformed, polygon_to_3D
    return transformed


def absolute_orientation(points_A, points_B, return_error=False):
    """
    Calculates the transform that best aligns points_A with points_B
    Uses Horn's method for the absolute orientation problem, in 3D with no scaling.
    :param points_A: nx3 list
    :param points_B: nx3 list
    :param return_error: boolean, if True returns 1xn list of euclidean distances representing the linear_distance from
            T*points_A[i] to points_B[i]
    :return: M: 4x4 transformation matrix for the transform that best aligns points_A to points_B， error: float,
                list of maximum euclidean linear_distance
    author: revised by weiwei
    date: 20201202
    """
    points_A = np.array(points_A)
    points_B = np.array(points_B)
    if (points_A.shape != points_B.shape):
        raise ValueError('Points must be of the same shape!')
    if len(points_A.shape) != 2 or points_A.shape[1] != 3:
        raise ValueError('Points must be (n,3)!')
    lc = np.average(points_A, axis=0)
    rc = np.average(points_B, axis=0)
    left = points_A - lc
    right = points_B - rc
    M = np.dot(left.T, right)
    [[Sxx, Sxy, Sxz],
     [Syx, Syy, Syz],
     [Szx, Szy, Szz]] = M
    N = [[(Sxx + Syy + Szz), (Syz - Szy), (Szx - Sxz), (Sxy - Syx)],
         [(Syz - Szy), (Sxx - Syy - Szz), (Sxy + Syx), (Szx + Sxz)],
         [(Szx - Sxz), (Sxy + Syx), (-Sxx + Syy - Szz), (Syz + Szy)],
         [(Sxy - Syx), (Szx + Sxz), (Syz + Szy), (-Sxx - Syy + Szz)]]
    (w, v) = np.linalg.eig(N)
    q = v[:, np.argmax(w)]
    q = q / np.linalg.norm(q)
    M1 = [[q[0], -q[1], -q[2], -q[3]],
          [q[1], q[0], q[3], -q[2]],
          [q[2], -q[3], q[0], q[1]],
          [q[3], q[2], -q[1], q[0]]]
    M2 = [[q[0], -q[1], -q[2], -q[3]],
          [q[1], q[0], -q[3], q[2]],
          [q[2], q[3], q[0], -q[1]],
          [q[3], -q[2], q[1], q[0]]]
    R = np.dot(np.transpose(M1), M2)[1:4, 1:4]
    T = rc - np.dot(R, lc)
    M = np.eye(4)
    M[0:3, 0:3] = R
    M[0:3, 3] = T
    if return_error:
        errors = np.sum((transform_points(points_A, M) - points_B) ** 2, axis=1)
        return M, errors.max()
    return M


def remove_close_pairs(points, radius):
    """
    Given an nxd set of points where d=2or3 return a list of points where no point is closer than major_radius
    :param points: a nxd list of points
    :param radius:
    :return:
    author: revised by weiwei
    date: 20201202
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    # get the index of every pair of points closer than our major_radius
    pairs = tree.query_pairs(radius, output_type='ndarray')
    # how often each vertex index appears in a pair
    # this is essentially a cheaply computed "vertex degree"
    # in the graph that we could construct for connected points
    count = np.bincount(pairs.ravel(), minlength=len(points))
    # for every pair we know we have to remove one of them
    # which of the two options we pick can have a large impact
    # on how much over-culling we end_type up doing
    column = count[pairs].argmax(axis=1)
    # take the value in each row with the highest degree
    # there is probably better numpy slicing you could do here
    highest = pairs.ravel()[column + 2 * np.arange(len(column))]
    # mask the vertices by index
    mask = np.ones(len(points), dtype=np.bool_)
    mask[highest] = False
    if tol.strict:
        # verify we actually did what we said we'd do
        test = cKDTree(points[mask])
        assert len(test.query_pairs(radius)) == 0
    return points[mask], mask


def remove_close_withfaceid(points, face_index, radius):
    """
    Given an nxd set of points where d=2or3 return a list of points where no point is closer than major_radius
    :param points:
    :param face_index:
    :param radius:
    :return:
    author: revised by weiwei
    date: 20201202
    """
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(points)
    consumed = np.zeros(len(points), dtype=np.bool_)
    unique = np.zeros(len(points), dtype=np.bool_)
    for i in range(len(points)):
        if consumed[i]: continue
        neighbors = tree.query_ball_point(points[i], r=radius)
        consumed[neighbors] = True
        unique[i] = True
    return points[unique], face_index[unique]


def remove_close_between_two_sets(points_fixed, points_reduce, radius):
    """
    Given two sets of points and a major_radius, return a set of points that is the subset of points_reduce where no point is
    within major_radius of any point in points_fixed
    author: revised by weiwei
    date: 20201202
    """
    from scipy.spatial import cKDTree as KDTree
    tree_fixed = KDTree(points_fixed)
    tree_reduce = KDTree(points_reduce)
    reduce_duplicates = tree_fixed.query_ball_tree(tree_reduce, r=radius)
    reduce_duplicates = np.unique(np.hstack(reduce_duplicates).astype(int))
    reduce_mask = np.ones(len(points_reduce), dtype=np.bool_)
    reduce_mask[reduce_duplicates] = False
    points_clean = points_reduce[reduce_mask]
    return points_clean


def k_means(points, k, **kwargs):
    """
    Find k centroids that attempt to minimize the k- means problem: https://en.wikipedia.org/wiki/Metric_k-center
    :param: points: nxd list of points
    :param: k: int, number of centroids to compute
    :param: **kwargs: passed directly to scipy.cluster.vq.kmeans
    :return: centroids: kxd list of points,m labels: 1xn list of indices for which points belong to which centroid
    author: revised by weiwei
    date: 20201202
    """
    from scipy.cluster.vq import kmeans
    from scipy.spatial import cKDTree
    points = np.asanyarray(points)
    points_std = points.std(axis=0)
    whitened = points / points_std
    centroids_whitened, distortion = kmeans(whitened, k, **kwargs)
    centroids = centroids_whitened * points_std
    tree = cKDTree(centroids)
    labels = tree.query(points, k=1)[1]
    return centroids, labels


def plot_points(points, show=True):
    import matplotlib.pyplot as plt
    points = np.asanyarray(points)
    dimension = points.shape[1]
    if dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*points.T)
    elif dimension == 2:
        plt.scatter(*points.T)
    else:
        raise ValueError('Points must be 2D or 3D, not %dD', dimension)
    if show:
        plt.show()
