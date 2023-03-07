import numpy as np

from .util import diagonal_dot, unitize, is_shape
from .points import point_plane_distance
from .constants import tol


def cross(triangles):
    '''
    Returns the cross product of two edges from input triangles 

    triangles: vertices of triangles (n,3,3)
    returns:   cross product of two edge vectors (n,3)
    '''
    vectors = np.diff(triangles, axis=1)
    crosses = np.cross(vectors[:, 0], vectors[:, 1])
    return crosses


def area(triangles, sum=True):
    '''
    Calculates the sum area of input triangles 

    Arguments
    ----------
    triangles: vertices of triangles (n,3,3)
    sum:       bool, return summed area or individual triangle area
    
    Returns
    ---------- 
    area:
        if sum: float, sum area of triangles
        else:   (n,) float, individual area of triangles
    '''
    crosses = cross(triangles)
    area = (np.sum(crosses ** 2, axis=1) ** .5) * .5
    if sum:
        return np.sum(area)
    return area


def normals(triangles):
    '''
    Calculates the normals of input triangles 
    
    triangles: vertices of triangles, (n,3,3)
    returns:   normal vectors, (n,3)
    '''
    crosses = cross(triangles)
    normals, valid = unitize(crosses, check_valid=True)
    return normals, valid


def all_coplanar(triangles):
    '''
    Given a list of triangles, return True if they are all coplanar, and False if not.
  
    triangles: vertices of triangles, (n,3,3)
    returns:   all_coplanar, bool
    '''
    test_normal = normals(triangles)[0]
    test_vertex = triangles[0][0]
    distances = point_plane_distance(points=triangles[1:].reshape((-1, 3)),
                                     plane_normal=test_normal,
                                     plane_origin=test_vertex)
    all_coplanar = np.all(np.abs(distances) < tol.zero)
    return all_coplanar


def any_coplanar(triangles):
    '''
    Given a list of triangles, if the FIRST triangle is coplanar with ANY
    of the following triangles, return True.
    Otherwise, return False. 
    '''
    test_normal = normals(triangles)[0]
    test_vertex = triangles[0][0]
    distances = point_plane_distance(points=triangles[1:].reshape((-1, 3)),
                                     plane_normal=test_normal,
                                     plane_origin=test_vertex)
    any_coplanar = np.any(np.all(np.abs(distances.reshape((-1, 3)) < tol.zero), axis=1))
    return any_coplanar


def mass_properties(triangles, density=1.0, skip_inertia=False):
    '''
    Calculate the mass properties of a group of triangles.
    
    Implemented from:
    http://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    '''
    crosses = cross(triangles)
    surface_area = np.sum(np.sum(crosses ** 2, axis=1) ** .5) * .5

    # these are the subexpressions of the integral 
    f1 = triangles.sum(axis=1)

    # triangles[:,0,:] will give rows like [[x0, y0, z0], ...] (the first vertex of every triangle)
    # triangles[:,:,0] will give rows like [[x0, x1, x2], ...] (the x coordinates of every triangle)
    f2 = (triangles[:, 0, :] ** 2 +
          triangles[:, 1, :] ** 2 +
          triangles[:, 0, :] * triangles[:, 1, :] +
          triangles[:, 2, :] * f1)

    f3 = ((triangles[:, 0, :] ** 3) +
          (triangles[:, 0, :] ** 2) * (triangles[:, 1, :]) +
          (triangles[:, 0, :]) * (triangles[:, 1, :] ** 2) +
          (triangles[:, 1, :] ** 3) +
          (triangles[:, 2, :] * f2))

    g0 = (f2 + (triangles[:, 0, :] + f1) * triangles[:, 0, :])
    g1 = (f2 + (triangles[:, 1, :] + f1) * triangles[:, 1, :])
    g2 = (f2 + (triangles[:, 2, :] + f1) * triangles[:, 2, :])

    integral = np.zeros((10, len(f1)))
    integral[0] = crosses[:, 0] * f1[:, 0]
    integral[1:4] = (crosses * f2).T
    integral[4:7] = (crosses * f3).T

    for i in range(3):
        triangle_i = np.mod(i + 1, 3)
        integral[i + 7] = crosses[:, i] * ((triangles[:, 0, triangle_i] * g0[:, i]) +
                                           (triangles[:, 1, triangle_i] * g1[:, i]) +
                                           (triangles[:, 2, triangle_i] * g2[:, i]))

    coefficents = 1.0 / np.array([6, 24, 24, 24, 60, 60, 60, 120, 120, 120])
    integrated = integral.sum(axis=1) * coefficents

    volume = integrated[0]
    center_mass = integrated[1:4] / volume

    result = {'density': density,
              'surface_area': surface_area,
              'volume': volume,
              'mass': density * volume,
              'center_mass': center_mass.tolist()}

    if skip_inertia:
        return result

    inertia = np.zeros((3, 3))
    inertia[0, 0] = integrated[5] + integrated[6] - (volume * (center_mass[[1, 2]] ** 2).sum())
    inertia[1, 1] = integrated[4] + integrated[6] - (volume * (center_mass[[0, 2]] ** 2).sum())
    inertia[2, 2] = integrated[4] + integrated[5] - (volume * (center_mass[[0, 1]] ** 2).sum())
    inertia[0, 1] = (integrated[7] - (volume * np.product(center_mass[[0, 1]])))
    inertia[1, 2] = (integrated[8] - (volume * np.product(center_mass[[1, 2]])))
    inertia[0, 2] = (integrated[9] - (volume * np.product(center_mass[[0, 2]])))
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    inertia[1, 0] = inertia[0, 1]
    inertia *= density

    result['inertia'] = inertia.tolist()

    return result


def windings_aligned(triangles, normals_compare):
    '''
    Given a set of triangles and a set of normals determine if the two are aligned
    
    Arguments
    ----------
    triangles: (n,3,3) set of vertex locations
    normals_compare: (n,3) set of normals

    Returns
    ----------
    aligned: (n) bool list, are normals aligned with triangles
    '''

    calculated, valid = normals(triangles)
    difference = diagonal_dot(calculated, normals_compare[valid])
    result = np.zeros(len(triangles), dtype=bool)
    result[valid] = difference > 0.0
    return result


def bounds_tree(triangles):
    '''
    Given a set of triangles, create an r-tree for broad- phase 
    collision detection

    Arguments
    ---------
    triangles: (n, 3, 3) list of vertices

    Returns
    ---------
    tree: Rtree object 
    '''
    from rtree import index

    # the property object required to get a 3D r-tree index
    properties = index.Property()
    properties.dimension = 3
    # the (n,6) interleaved bounding box for every triangle
    tri_bounds = np.column_stack((triangles.min(axis=1), triangles.max(axis=1)))

    # stream loading wasn't getting proper index
    tree = index.Index(properties=properties)
    for i, bounds in enumerate(tri_bounds):
        tree.insert(i, bounds)
    return tree


def nondegenerate(triangles):
    '''
    Find all triangles which have nonzero area.
    Degenerate triangles are divided into two cases:
    1) Two of the three vertices are colocated
    2) All three vertices are unique but colinear

    Arguments
    ----------
    triangles: (n, 3, 3) float, set of triangles

    Returns
    ----------
    nondegenerate: (n,) bool array of triangles that have area
    '''
    a, valid_a = unitize(triangles[:, 1] - triangles[:, 0], check_valid=True)
    b, valid_b = unitize(triangles[:, 2] - triangles[:, 0], check_valid=True)

    diff = np.zeros((len(triangles), 3))
    diff[valid_a] = a
    diff[valid_b] -= b

    ok = (np.abs(diff) > tol.merge).any(axis=1)
    ok[np.logical_not(valid_a)] = False
    ok[np.logical_not(valid_b)] = False

    return ok


def closest_point(triangles, points):
    """
    Return the closest point on the surface of each triangle for a
    list of corresponding points.
    Implements the method from "Real Time Collision Detection" and
    use the same variable names as "ClosestPtPointTriangle" to avoid
    being any more confusing.
    Parameters
    ----------
    triangles : (n, 3, 3) float
      Triangle vertices in space
    points : (n, 3) float
      Points in space
    Returns
    ----------
    closest : (n, 3) float
      Point on each triangle closest to each point
    """

    # check input triangles and points
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(triangles, (-1, 3, 3)):
        raise ValueError('triangles shape incorrect')
    if not util.is_shape(points, (len(triangles), 3)):
        raise ValueError('need same number of triangles and points!')

    # store the location of the closest point
    result = np.zeros_like(points)
    # which points still need to be handled
    remain = np.ones(len(points), dtype=np.bool)

    # if we dot product this against a (n, 3)
    # it is equivalent but faster than array.sum(axis=1)
    ones = [1.0, 1.0, 1.0]

    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]

    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = points - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = np.dot(ab * ap, ones)
    d2 = np.dot(ac * ap, ones)

    # is the point at A
    is_a = np.logical_and(d1 < tol.zero, d2 < tol.zero)
    if any(is_a):
        result[is_a] = a[is_a]
        remain[is_a] = False

    # check if P in vertex region outside B
    bp = points - b
    d3 = np.dot(ab * bp, ones)
    d4 = np.dot(ac * bp, ones)

    # do the logic check
    is_b = (d3 > -tol.zero) & (d4 <= d3) & remain
    if any(is_b):
        result[is_b] = b[is_b]
        remain[is_b] = False

    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < tol.zero) &
             (d1 > -tol.zero) &
             (d3 < tol.zero) & remain)
    if any(is_ab):
        v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).reshape((-1, 1))
        result[is_ab] = a[is_ab] + (v * ab[is_ab])
        remain[is_ab] = False

    # check if P in vertex region outside C
    cp = points - c
    d5 = np.dot(ab * cp, ones)
    d6 = np.dot(ac * cp, ones)
    is_c = (d6 > -tol.zero) & (d5 <= d6) & remain
    if any(is_c):
        result[is_c] = c[is_c]
        remain[is_c] = False

    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < tol.zero) & (d2 > -tol.zero) & (d6 < tol.zero) & remain
    if any(is_ac):
        w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).reshape((-1, 1))
        result[is_ac] = a[is_ac] + w * ac[is_ac]
        remain[is_ac] = False

    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < tol.zero) &
             ((d4 - d3) > - tol.zero) &
             ((d5 - d6) > -tol.zero) & remain)
    if any(is_bc):
        d43 = d4[is_bc] - d3[is_bc]
        w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).reshape((-1, 1))
        result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
        remain[is_bc] = False

    # any remaining points must be inside face region
    if any(remain):
        # point is inside face region
        denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
        v = (vb[remain] * denom).reshape((-1, 1))
        w = (vc[remain] * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)

    return result
