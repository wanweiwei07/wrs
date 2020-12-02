"""
Generate a bunch of trimesh objects, in meter radian
"""

import math
import numpy as np
from basics import robotmath as rm
import trimesh.primitives as tp
import trimesh as trm
from shapely.geometry import Polygon


def genbox(extent=np.array([1, 1, 1]), homomat=np.eye(4)):
    """
    :param extent: x, y, z (origin is 0)
    :param homomat: rotation and translation
    :return: a Trimesh object (Primitive)
    author: weiwei
    date: 20191228osaka
    """
    return tp.Box(box_extents=extent, box_transform=homomat)


def genstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, type="rect", sections=8):
    """
    interface to genrectstick/genroundstick
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param type: simple or smooth
    :param sections: # of discretized sectors used to approximate a cylinder
    :return:
    author: weiwei
    date: 20191228osaka
    """
    if type is "rect":
        return genrectstick(spos, epos, thickness, sections=sections)
    if type is "round":
        return genroundstick(spos, epos, thickness, count=[sections / 2.0, sections / 2.0])


def genrectstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=.005, sections=8):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param sections: # of discretized sectors used to approximate a cylinder
    :return: a Trimesh object (Primitive)
    author: weiwei
    date: 20191228osaka
    """
    pos = spos
    height = np.linalg.norm(epos - spos)
    if np.allclose(height, 0):
        rotmat = np.eye(3)
    else:
        rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    return tp.Cylinder(height=height, radius=thickness / 2.0, sections=sections, homomat=homomat)


def genroundstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, count=[8, 8]):
    """
    :param spos:
    :param epos:
    :param thickness:
    :return: a Trimesh object (Primitive)
    author: weiwei
    date: 20191228osaka
    """
    pos = spos
    height = np.linalg.norm(epos - spos)
    if np.allclose(height, 0):
        rotmat = np.eye(3)
    else:
        rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    return tp.Capsule(height=height, radius=thickness / 2.0, count=count, homomat=homomat)


def gensphere(pos=np.array([0, 0, 0]), radius=0.02, subdivisions=2):
    """
    :param pos: 1x3 nparray
    :param radius: 0.02 m by default
    :param subdivisions: levels of icosphere discretization
    :return:
    author: weiwei
    date: 20191228osaka
    """
    return tp.Sphere(sphere_radius=radius, sphere_center=pos, subdivisions=subdivisions)


def genellipsoid(pos=np.array([0, 0, 0]), axmat=np.eye(3), subdivisions=5):
    """
    :param pos:
    :param axmat: 3x3 mat, each column is an axis of the ellipse
    :param subdivisions: levels of icosphere discretization
    :return:
    author: weiwei
    date: 20191228osaka
    """
    homomat = rm.homomat_from_posrot(pos, axmat)
    sphere = tp.Sphere(sphere_radius=1, sphere_center=pos, subdivisions=subdivisions)
    vertices = rm.homomat_transform_points(homomat, sphere.vertices)
    return trm.Trimesh(vertices=vertices, faces=sphere.faces)


def gendumbbell(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, sections=8, subdivisions=1):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param sections:
    :param subdivisions: levels of icosphere discretization
    :return:
    author: weiwei
    date: 20191228osaka
    """
    stick = genrectstick(spos=spos, epos=epos, thickness=thickness, sections=sections)
    sposball = gensphere(pos=spos, radius=thickness, subdivisions=subdivisions)
    endball = gensphere(pos=epos, radius=thickness, subdivisions=subdivisions)
    vertices = np.vstack((stick.vertices, sposball.vertices, endball.vertices))
    sposballfaces = sposball.faces + len(stick.vertices)
    endballfaces = endball.faces + len(sposball.vertices) + len(stick.vertices)
    faces = np.vstack((stick.faces, sposballfaces, endballfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gencone(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), radius=0.005, sections=8):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param sections: # of discretized sectors used to approximate a cylinder
    :return:
    author: weiwei
    date: 20191228osaka
    """
    height = np.linalg.norm(spos - epos)
    pos = spos
    rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    return tp.Cone(height=height, radius=radius, sections=sections, homomat=homomat)


def genarrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, sections=8, sticktype="rect"):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param sections: # of discretized sectors used to approximate a cylinder
    :param sticktype: The shape at the end of the arrow stick, round or rect
    :param radius:
    :return:
    author: weiwei
    date: 20191228osaka
    """
    direction = rm.unit_vector(epos - spos)
    stick = genstick(spos=spos, epos=epos - direction * thickness * 4, thickness=thickness, type=sticktype,
                     sections=sections)
    cap = gencone(spos=epos - direction * thickness * 4, epos=epos, radius=thickness, sections=sections)
    vertices = np.vstack((stick.vertices, cap.vertices))
    capfaces = cap.faces + len(stick.vertices)
    faces = np.vstack((stick.faces, capfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gendasharrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, lsolid=None, lspace=None,
                 sections=8, sticktype="rect"):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param lsolid: length of the solid section, 1*thickness if None
    :param lspace: length of the empty section, 1.5*thickness if None
    :return:
    author: weiwei
    date: 20191228osaka
    """
    solidweight = 1.6
    spaceweight = 1.07
    totalweight = solidweight + spaceweight
    if not lsolid:
        lsolid = thickness * solidweight
    if not lspace:
        lspace = thickness * spaceweight
    length, direction = rm.unit_vector(epos - spos, togglelength=True)
    nstick = math.floor(length / (thickness * totalweight))
    cap = gencone(spos=epos - direction * thickness * 4, epos=epos, radius=thickness, sections=sections)
    stick = genstick(spos=epos - direction * thickness * 4 - lsolid * direction, epos=epos - direction * thickness * 4,
                     thickness=thickness, type=sticktype, sections=sections)
    vertices = np.vstack((cap.vertices, stick.vertices))
    stickfaces = stick.faces + len(cap.vertices)
    faces = np.vstack((cap.faces, stickfaces))
    for i in range(1, nstick - 1):
        tmpspos = epos - direction * thickness * 4 - lsolid * direction - (lsolid * direction + lspace * direction) * i
        tmpstick = genstick(spos=tmpspos, epos=tmpspos + lsolid * direction, thickness=thickness, type=sticktype,
                            sections=sections)
        tmpstickfaces = tmpstick.faces + len(vertices)
        vertices = np.vstack((vertices, tmpstick.vertices))
        faces = np.vstack((faces, tmpstickfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def genaxis(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=0.1, thickness=0.005):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :return:
    author: weiwei
    date: 20191228osaka
    """
    directionx = rotmat[:, 0]
    directiony = rotmat[:, 1]
    directionz = rotmat[:, 2]
    # x
    endx = directionx * length
    stickx = genstick(spos=pos, epos=endx, thickness=thickness)
    capx = gencone(spos=endx, epos=endx + directionx * thickness * 4, radius=thickness)
    # y
    endy = directiony * length
    sticky = genstick(spos=pos, epos=endy, thickness=thickness)
    capy = gencone(spos=endy, epos=endy + directiony * thickness * 4, radius=thickness)
    # z
    endz = directionz * length
    stickz = genstick(spos=pos, epos=endz, thickness=thickness)
    capz = gencone(spos=endz, epos=endz + directionz * thickness * 4, radius=thickness)
    vertices = np.vstack(
        (stickx.vertices, capx.vertices, sticky.vertices, capy.vertices, stickz.vertices, capz.vertices))
    capxfaces = capx.faces + len(stickx.vertices)
    stickyfaces = sticky.faces + len(stickx.vertices) + len(capx.vertices)
    capyfaces = capy.faces + len(stickx.vertices) + len(capx.vertices) + len(sticky.vertices)
    stickzfaces = stickz.faces + len(stickx.vertices) + len(capx.vertices) + len(sticky.vertices) + len(capy.vertices)
    capzfaces = capz.faces + len(stickx.vertices) + len(capx.vertices) + len(sticky.vertices) + len(
        capy.vertices) + len(stickz.vertices)
    faces = np.vstack((stickx.faces, capxfaces, stickyfaces, capyfaces, stickzfaces, capzfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gentorus(axis=np.array([1, 0, 0]), portion=.5, center=np.array([0, 0, 0]), radius=0.005, thickness=0.0015,
             sections=8,
             discretization=24):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param radius:
    :param thickness:
    :param sections: # of discretized sectors used to approximate a cylindrical stick
    :param discretization: number sticks used for approximating a torus
    :return:
    author: weiwei
    date: 20200602
    """
    unitaxis = rm.unit_vector(axis)
    startingaxis = rm.orthogonal_vector(unitaxis)
    startingpos = startingaxis * radius + center
    discretizedangle = 2 * math.pi / discretization
    ndist = int(portion * discretization)
    # gen the last sec first
    # gen the remaining torus afterwards
    if ndist > 0:
        lastpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, (ndist - 1) * discretizedangle),
                                  startingaxis) * radius
        nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, ndist * discretizedangle), startingaxis) * radius
        stick = genstick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
        vertices = stick.vertices
        faces = stick.faces
        lastpos = startingpos
        for i in range(1 * np.sign(ndist), ndist, 1 * np.sign(ndist)):
            nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, i * discretizedangle), startingaxis) * radius
            stick = genstick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
            stickfaces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stickfaces))
            lastpos = nxtpos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        return trm.Trimesh()


def gencircarrow(axis=np.array([1, 0, 0]), portion=0.3, center=np.array([0, 0, 0]), radius=0.005, thickness=0.0015,
                 sections=8, discretization=24):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param radius:
    :param thickness:
    :param rgba:
    :param discretization: number sticks used for approximation
    :return:
    author: weiwei
    date: 20200602
    """
    unitaxis = rm.unit_vector(axis)
    startingaxis = rm.orthogonal_vector(unitaxis)
    startingpos = startingaxis * radius + center
    discretizedangle = 2 * math.pi / discretization
    ndist = int(portion * discretization)
    # gen the last arrow first
    # gen the remaining torus
    if ndist > 0:
        lastpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, (ndist - 1) * discretizedangle),
                                  startingaxis) * radius
        nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, ndist * discretizedangle), startingaxis) * radius
        arrow = genarrow(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, sticktype="round")
        vertices = arrow.vertices
        faces = arrow.faces
        lastpos = startingpos
        for i in range(1 * np.sign(ndist), ndist, 1 * np.sign(ndist)):
            nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, i * discretizedangle), startingaxis) * radius
            stick = genstick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
            stickfaces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stickfaces))
            lastpos = nxtpos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        return trm.Trimesh()


def facetboundary(objtrimesh, facet, facetcenter, facetnormal):
    """
    compute a boundary polygon for facet
    assumptions:
    1. there is only one boundary
    2. the facet is convex
    :param objtrimesh: a datatype defined in trimesh
    :param facet: a data type defined in trimesh
    :param facetcenter and facetnormal used to compute the transform, see trimesh.geometry.plane_transform
    :return: [1x3 vertices list, 1x2 vertices list, 4x4 homogeneous transformation matrix)]
    author: weiwei
    date: 20161213tsukuba
    """
    facetp = None
    # use -facetnormal to let the it face downward
    facethomomat = trm.geometry.plane_transform(facetcenter, -facetnormal)
    for i, faceidx in enumerate(facet):
        vert0 = objtrimesh.vertices[objtrimesh.faces[faceidx][0]]
        vert1 = objtrimesh.vertices[objtrimesh.faces[faceidx][1]]
        vert2 = objtrimesh.vertices[objtrimesh.faces[faceidx][2]]
        vert0p = rm.homotransformpoint(facethomomat, vert0)
        vert1p = rm.homotransformpoint(facethomomat, vert1)
        vert2p = rm.homotransformpoint(facethomomat, vert2)
        facep = Polygon([vert0p[:2], vert1p[:2], vert2p[:2]])
        if facetp is None:
            facetp = facep
        else:
            facetp = facetp.union(facep)
    verts2d = list(facetp.exterior.coords)
    verts3d = []
    for vert2d in verts2d:
        vert3d = rm.homotransformpoint(rm.homoinverse(facethomomat), np.array([vert2d[0], vert2d[1], 0]))[:3]
        verts3d.append(vert3d)
    return verts3d, verts2d, facethomomat


if __name__ == "__main__":
    from visualization.panda import world

    base = world.World(camp=[0, .3, 0], lookatpos=[0, 0, 0], autocamrotate=True)
    import environment.collisionmodel as cm

    objcm = cm.CollisionModel(gentorus())
    # objcm.reparentTo(base.render)
    # objcm = cm.CollisionModel(genrectstick(thickness=5))
    # objcm.setColor(1,0,0,1)
    # objcm.reparentTo(base.render)
    # objcm = cm.CollisionModel(gendumbbell())
    objcm.setColor(1, 0, 0, 1)
    objcm.reparentTo(base.render)
    # base.pggen.genArrow(length=100, thickness=5).reparentTo(base.render)

    base.run()
