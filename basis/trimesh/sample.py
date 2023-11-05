import numpy as np
from . import util
from . import transformations


def sample_surface(mesh, count):
    """
    Sample the surface of a mesh, returning the specified number of points
    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    :param mesh: a Trimesh instance
    :param count: number of points to return
    :return:
    author: revised by weiwei
    date: 20200120
    """
    # len(mesh.faces) float array of the areas of each face of the mesh
    area = mesh.area_faces
    # total area (float)
    area_sum = np.sum(area)
    # cumulative area (len(mesh.faces)) 
    area_cum = np.cumsum(area)
    face_pick = np.random.random(count) * area_sum
    face_index = np.searchsorted(area_cum, face_pick)
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.triangles[:, 0]
    tri_vectors = mesh.triangles[:, 1:].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]
    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = np.random.random((len(tri_vectors), 2, 1))
    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)
    # multiply triangle edge vectors by the random lengths and sum
    points_vector = (tri_vectors * random_lengths).sum(axis=1)
    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    points = points_vector + tri_origins
    return points, face_index


# def sample_surface_withfaceid(mesh, n_sec_minor):
#     '''
#     Sample the surface of a mesh, returning the specified number of points
#
#     For individual triangle sampling uses this method:
#     http://mathworld.wolfram.com/TrianglePointPicking.html
#
#     Arguments
#     ---------
#     mesh: Trimesh object
#     n_sec_minor: number of points to return
#
#     Returns
#     ---------
#     samples: (n_sec_minor,3) points in space on the surface of mesh
#
#     '''
#
#     # len(mesh.faces) float array of the areas of each face of the mesh
#     area = mesh.area_faces
#     # total area (float)
#     area_sum = np.sum(area)
#     # cumulative area (len(mesh.faces))
#     area_cum = np.cumsum(area)
#     face_pick = np.random.random(n_sec_minor) * area_sum
#     face_index = np.searchsorted(area_cum, face_pick)
#
#     # pull triangles into the form of an origin + 2 vectors
#     tri_origins = mesh.triangles[:, 0]
#     tri_vectors = mesh.triangles[:, 1:].copy()
#     tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))
#
#     # pull the vectors for the faces we are going to sample from
#     tri_origins = tri_origins[face_index]
#     tri_vectors = tri_vectors[face_index]
#
#     # randomly generate two 0-1 scalar components to multiply edge vectors by
#     random_lengths = np.random.random((len(tri_vectors), 2, 1))
#
#     # points will be distributed on a quadrilateral if we use 2 0-1 samples
#     # if the two scalar components sum less than 1.0 the point will be
#     # inside the triangle, so we find vectors longer than 1.0 and
#     # transform them to be inside the triangle
#     random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
#     random_lengths[random_test] -= 1.0
#     random_lengths = np.abs(random_lengths)
#
#     # multiply triangle edge vectors by the random lengths and sum
#     sample_vector = (tri_vectors * random_lengths).sum(axis=1)
#
#     # finally, offset by the origin to generate
#     # (n,3) points in space on the triangle
#     samples = sample_vector + tri_origins
#
#     return samples, face_index


def sample_volume(mesh, count):
    """
    Use rejection sampling to produce points randomly
    distributed in the volume of a mesh
    :param mesh:
    :param count:
    :return:
    author: revised by weiwei
    date: 20210120
    """
    points = (np.random.random((count, 3)) * mesh.extents) + mesh.bounds[0]
    contained = mesh.contains(points)
    samples = points[contained][:count]
    return samples


def sample_box_volume(extents,
                      count,
                      transform=None):
    """
    Use rejection sampling to produce points randomly
    distributed in the volume of a given box
    :param extents: 1x3 nparray
    :param count: npoints
    :param transform: homogeneous transformation matrix
    :return: nx3 points in the requested volume
    author: revised by weiwei
    date: 20210120
    """
    samples = np.random.random((count, 3)) - .5
    samples *= extents
    if transform is not None:
        samples = transformations.transform_points(samples,
                                                   transform)
    return samples


def sample_surface_even(mesh, count, radius=None):
    """
    Sample the surface of a mesh, returning samples which are
    approximately evenly spaced.
    Note that since it is using rejection sampling it may return
    fewer points than requested (i.e. n < n_sec_minor). If this is the
    case a log.warning will be emitted.
    :param mesh:
    :param count:
    :param radius:
    :return:
    author: revised by weiwei
    date: 20210120
    """
    from .points import remove_close_withfaceid
    # guess major_radius from area
    if radius is None:
        radius = np.sqrt(mesh.area / (3 * count))
    # get points on the surface
    points, index = sample_surface(mesh, count * 3)
    # remove the points closer than major_radius
    points, index = remove_close_withfaceid(points, index, radius)
    # we got all the samples we expect
    if len(points) >= count:
        return points[:count], index[:count]
    # warn if we didn't get all the samples we expect
    # util.log.warning('only got {}/{} samples!'.format(len(points), n_sec_minor)) TODO
    return points, index


def sample_surface_sphere(count):
    """
    Correctly pick random points on the surface of a unit sphere
    Uses this method:
    http://mathworld.wolfram.com/SpherePointPicking.html
    :param count:
    :return nx3 points on a unit sphere
    """
    # get random values 0.0-1.0
    u, v = np.random.random((2, count))
    # convert to two angles
    theta = np.pi * 2 * u
    phi = np.arccos((2 * v) - 1)
    # convert spherical coordinates to cartesian
    points = util.spherical_to_vector(np.column_stack((theta, phi)))
    return points
