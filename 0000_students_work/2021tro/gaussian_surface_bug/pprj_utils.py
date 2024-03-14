import numpy as np
import matplotlib.pyplot as plt
import pandaplotutils.pandactrl as pc
import pickle
import matplotlib.pyplot as plt


def psp_mat4(fovy, aspect, n, f):
    """

    :param fovy: vertical field of view
    :param aspect: aspect ratio
    :param n: near Z Plane
    :param f: far Z Plane
    :return:
    """
    q = 1.0 / np.tan(np.radians(0.5 * fovy))
    A = q / aspect
    B = (n + f) / (n - f)
    C = (2.0 * n * f) / (n - f)

    result = np.asarray([[A, 0.0, 0.0, 0.0],
                         [0.0, q, 0.0, 0.0],
                         [0.0, 0.0, B, -1.0],
                         [0.0, 0.0, C, 0.0]])

    return result


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2

    Args
    ----
        vec1 (numpy.ndarray): A 3d "source" vector
        vec2 (numpy.ndarray): A 3d "destination" vector

    Returns
    -------
        numpy.ndarray: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def trans_projection(ps, plane_n, view_p, view_plane_dist):
    """
    Project the given 3D points in world coordinate on to a given plane.

    Args:
        ps (numpy.ndarray): 3D points in world coordinate
        plane_n (numpy.ndarray): normal motion_vec of the target plane
        view_p (numpy.ndarray): view point location
        view_plane_dist (float): linear_distance from view point to the target plane

    Returns:
        numpy.ndarray: uv coordinates on the plane
    """

    ps = np.asarray(ps)
    plane_n = np.asarray(plane_n)
    view_p = np.asarray(view_p)

    # step 1: calculate camera extrinsic matrix
    translation_w2c = view_p
    rotation_w2c = rotation_matrix_from_vectors(np.array([1, 0, 0]), plane_n)
    extrinsic_mat = np.vstack(
        [np.concatenate((rotation_w2c, translation_w2c.reshape(-1, 1)), axis=1), [0, 0, 0, 1]]
    )

    # step 2: transform points from world coord to camera coord
    homo_points = ps
    homo_points = np.column_stack((homo_points, [1] * homo_points.shape[0]))
    for i in range(homo_points.shape[0]):
        homo_points[i] = np.linalg.inv(extrinsic_mat).dot(homo_points[i])

    # step 3: calculate projected point on plane
    plane_points = []
    for hp in homo_points:
        ratio = view_plane_dist / abs(hp[0])
        u = - hp[1] * ratio
        v = - hp[2] * ratio
        plane_points.append([u, v])

    return np.array(plane_points)


if __name__ == '__main__':
    """
    set up env and param
    """
    # base = pc.World(camp=np.array([500, -1000, 1000]), lookat_pos=np.array([0, 0, 50]))
    base = pc.World(camp=np.array([0, 0, 1000]), lookatpos=np.array([0, 0, 50]))
    mat4 = psp_mat4(30, 1, 100, 1000)
    ps = [[0, 0, 0],
          [1, 0, 0],
          [0, 1, 0],
          [1, 1, 0]]
    for p in ps:
        base.pggen.plotSphere(base.render, p[:3], rgba=(1, 0, 0, 1))
    plane_n = np.asarray([1, 0, -1])
    view_p = np.asarray([-10, 0, 10])
    view_plane_dist = 3

    pts = trans_projection(ps, plane_n, view_p, view_plane_dist=3)
    base.pggen.plotArrow(base.render, spos=view_p, epos=view_p + view_plane_dist * plane_n, rgba=(1, 0, 0, 1),
                         thickness=1)

    for pt in pts:
        base.pggen.plotSphere(base.render, (pt[0], pt[1], view_p[2]), rgba=(0, 1, 0, 1))

    print(pts)
    plt.scatter(pts[:, 0], pts[:, 1])
    plt.show()

    base.run()
