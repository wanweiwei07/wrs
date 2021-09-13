"""
Generate a bunch of trimesh objects, in meter radian
"""

import math
import numpy as np
import basis.trimesh.primitives as tp
import basis.trimesh as trm
import basis.robot_math as rm
import shapely.geometry as shpg


def gen_box(extent=np.array([1, 1, 1]), homomat=np.eye(4)):
    """
    :param extent: x, y, z (origin is 0)
    :param homomat: rotation and translation
    :return: a Trimesh object (Primitive)
    author: weiwei
    date: 20191228osaka
    """
    return tp.Box(box_extents=extent, box_transform=homomat)


def gen_stick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, type="rect", sections=8):
    """
    interface to genrectstick/genroundstick
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param type: rect or round
    :param sections: # of discretized sectors used to approximate a cylinder
    :return:
    author: weiwei
    date: 20191228osaka
    """
    if type == "rect":
        return gen_rectstick(spos, epos, thickness, sections=sections)
    if type == "round":
        return gen_roundstick(spos, epos, thickness, count=[sections / 2.0, sections / 2.0])


def gen_rectstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=.005, sections=8):
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


def gen_roundstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, count=[8, 8]):
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


def gen_dashstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, lsolid=None, lspace=None,
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
    if not lsolid:
        lsolid = thickness * solidweight
    if not lspace:
        lspace = thickness * spaceweight
    length, direction = rm.unit_vector(epos - spos, toggle_length=True)
    nstick = math.floor(length / (lsolid + lspace))
    vertices = np.empty((0, 3))
    faces = np.empty((0, 3))
    for i in range(0, nstick):
        tmp_spos = spos + (lsolid * direction + lspace * direction) * i
        tmp_stick = gen_stick(spos=tmp_spos,
                              epos=tmp_spos + lsolid * direction,
                              thickness=thickness,
                              type=sticktype,
                              sections=sections)
        tmp_stick_faces = tmp_stick.faces + len(vertices)
        vertices = np.vstack((vertices, tmp_stick.vertices))
        faces = np.vstack((faces, tmp_stick_faces))
    # wrap up the last segment
    tmp_spos = spos + (lsolid * direction + lspace * direction) * nstick
    tmp_epos = tmp_spos + lsolid * direction
    final_length, _ = rm.unit_vector(tmp_epos - spos, toggle_length=True)
    if final_length > length:
        tmp_epos = epos
    tmp_stick = gen_stick(spos=tmp_spos,
                          epos=tmp_epos,
                          thickness=thickness,
                          type=sticktype,
                          sections=sections)
    tmp_stick_faces = tmp_stick.faces + len(vertices)
    vertices = np.vstack((vertices, tmp_stick.vertices))
    faces = np.vstack((faces, tmp_stick_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.02, subdivisions=2):
    """
    :param pos: 1x3 nparray
    :param radius: 0.02 m by default
    :param subdivisions: levels of icosphere discretization
    :return:
    author: weiwei
    date: 20191228osaka
    """
    return tp.Sphere(sphere_radius=radius, sphere_center=pos, subdivisions=subdivisions)


def gen_ellipsoid(pos=np.array([0, 0, 0]), axmat=np.eye(3), subdivisions=5):
    """
    :param pos:
    :param axmat: 3x3 mat, each column is an axis of the ellipse
    :param subdivisions: levels of icosphere discretization
    :return:
    author: weiwei
    date: 20191228osaka
    """
    sphere = tp.Sphere(sphere_radius=1, sphere_center=np.zeros(3), subdivisions=subdivisions)
    vertices = axmat.dot(sphere.vertices.T).T
    vertices = vertices+pos
    return trm.Trimesh(vertices=vertices, faces=sphere.faces)


def gen_dumbbell(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, sections=8, subdivisions=1):
    """
    NOTE: return stick+spos_ball+epos_ball also work, but it is a bit slower
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param thickness: 0.005 m by default
    :param sections:
    :param subdivisions: levels of icosphere discretization
    :return:
    author: weiwei
    date: 20191228osaka
    """
    stick = gen_rectstick(spos=spos, epos=epos, thickness=thickness, sections=sections)
    spos_ball = gen_sphere(pos=spos, radius=thickness, subdivisions=subdivisions)
    epos_ball = gen_sphere(pos=epos, radius=thickness, subdivisions=subdivisions)
    vertices = np.vstack((stick.vertices, spos_ball.vertices, epos_ball.vertices))
    sposballfaces = spos_ball.faces + len(stick.vertices)
    endballfaces = epos_ball.faces + len(spos_ball.vertices) + len(stick.vertices)
    faces = np.vstack((stick.faces, sposballfaces, endballfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_cone(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), radius=0.005, sections=8):
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


def gen_arrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, sections=8, sticktype="rect"):
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
    stick = gen_stick(spos=spos, epos=epos - direction * thickness * 4, thickness=thickness, type=sticktype,
                      sections=sections)
    cap = gen_cone(spos=epos - direction * thickness * 4, epos=epos, radius=thickness, sections=sections)
    vertices = np.vstack((stick.vertices, cap.vertices))
    capfaces = cap.faces + len(stick.vertices)
    faces = np.vstack((stick.faces, capfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_dasharrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), thickness=0.005, lsolid=None, lspace=None,
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
    length, direction = rm.unit_vector(epos - spos, toggle_length=True)
    cap = gen_cone(spos=epos - direction * thickness * 4, epos=epos, radius=thickness, sections=sections)
    dash_stick = gen_dashstick(spos=spos,
                               epos=epos - direction * thickness * 4,
                               thickness=thickness,
                               lsolid=lsolid,
                               lspace=lspace,
                               sections=sections,
                               sticktype=sticktype)
    tmp_stick_faces = dash_stick.faces + len(cap.vertices)
    vertices = np.vstack((cap.vertices, dash_stick.vertices))
    faces = np.vstack((cap.faces, tmp_stick_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_axis(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=0.1, thickness=0.005):
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
    stickx = gen_stick(spos=pos, epos=endx, thickness=thickness)
    capx = gen_cone(spos=endx, epos=endx + directionx * thickness * 4, radius=thickness)
    # y
    endy = directiony * length
    sticky = gen_stick(spos=pos, epos=endy, thickness=thickness)
    capy = gen_cone(spos=endy, epos=endy + directiony * thickness * 4, radius=thickness)
    # z
    endz = directionz * length
    stickz = gen_stick(spos=pos, epos=endz, thickness=thickness)
    capz = gen_cone(spos=endz, epos=endz + directionz * thickness * 4, radius=thickness)
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


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=.5,
              center=np.array([0, 0, 0]),
              radius=0.1,
              thickness=0.005,
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
    if starting_vector is None:
        starting_vector = rm.orthogonal_vector(unitaxis)
    else:
        starting_vector = rm.unit_vector(starting_vector)
    starting_pos = starting_vector * radius + center
    discretizedangle = 2 * math.pi / discretization
    ndist = int(portion * discretization)
    # gen the last sec first
    # gen the remaining torus afterwards
    if ndist > 0:
        lastpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, (ndist - 1) * discretizedangle),
                                  starting_vector) * radius
        nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, ndist * discretizedangle), starting_vector) * radius
        stick = gen_stick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
        vertices = stick.vertices
        faces = stick.faces
        lastpos = starting_pos
        for i in range(1 * np.sign(ndist), ndist, 1 * np.sign(ndist)):
            nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, i * discretizedangle), starting_vector) * radius
            stick = gen_stick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
            stickfaces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stickfaces))
            lastpos = nxtpos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        return trm.Trimesh()


def gen_dashtorus(axis=np.array([1, 0, 0]),
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  radius=0.1,
                  thickness=0.005,
                  lsolid=None,
                  lspace=None,
                  sections=8,
                  discretization=24):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param radius:
    :param thickness:
    :param lsolid: length of solid
    :param lspace: length of space
    :param sections: # of discretized sectors used to approximate a cylindrical stick
    :param discretization: number sticks used for approximating a torus
    :return:
    author: weiwei
    date: 20200602
    """
    assert (0 <= portion <= 1)
    solidweight = 1.6
    spaceweight = 1.07
    if not lsolid:
        lsolid = thickness * solidweight
    if not lspace:
        lspace = thickness * spaceweight
    unit_axis = rm.unit_vector(axis)
    starting_vector = rm.orthogonal_vector(unit_axis)
    min_discretization_value = math.ceil(2 * math.pi / (lsolid + lspace))
    if discretization < min_discretization_value:
        discretization = min_discretization_value
    nsections = math.floor(portion * 2 * math.pi * radius / (lsolid + lspace))
    vertices = np.empty((0, 3))
    faces = np.empty((0, 3))
    for i in range(0, nsections):  # TODO wrap up end
        torus_sec = gen_torus(axis=axis,
                              starting_vector=rm.rotmat_from_axangle(axis, 2 * math.pi * portion / nsections * i).dot(
                                  starting_vector),
                              portion=portion / nsections * lsolid / (lsolid + lspace), center=center, radius=radius,
                              thickness=thickness, sections=sections, discretization=discretization)
        torus_sec_faces = torus_sec.faces + len(vertices)
        vertices = np.vstack((vertices, torus_sec.vertices))
        faces = np.vstack((faces, torus_sec_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=0.3,
                  center=np.array([0, 0, 0]),
                  radius=0.005,
                  thickness=0.0015,
                  sections=8,
                  discretization=24):
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
    if starting_vector is None:
        starting_vector = rm.orthogonal_vector(unitaxis)
    else:
        starting_vector = rm.unit_vector(starting_vector)
    starting_pos = starting_vector * radius + center
    discretizedangle = 2 * math.pi / discretization
    ndist = int(portion * discretization)
    # gen the last arrow first
    # gen the remaining torus
    if ndist > 0:
        lastpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, (ndist - 1) * discretizedangle),
                                  starting_vector) * radius
        nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, ndist * discretizedangle), starting_vector) * radius
        arrow = gen_arrow(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, sticktype="round")
        vertices = arrow.vertices
        faces = arrow.faces
        lastpos = starting_pos
        for i in range(1 * np.sign(ndist), ndist, 1 * np.sign(ndist)):
            nxtpos = center + np.dot(rm.rotmat_from_axangle(unitaxis, i * discretizedangle), starting_vector) * radius
            stick = gen_stick(spos=lastpos, epos=nxtpos, thickness=thickness, sections=sections, type="round")
            stickfaces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stickfaces))
            lastpos = nxtpos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        return trm.Trimesh()


def facet_boundary(objtrimesh, facet, facetcenter, facetnormal):
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
        facep = shpg.Polygon([vert0p[:2], vert1p[:2], vert2p[:2]])
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


def extract_subtrimesh(objtrm, face_id_list, offset_pos=np.zeros(3), offset_rotmat=np.eye(3)):
    """
    :param objtrm:
    :param face_id_list:
    :param offset_pos:
    :param offset_rotmat:
    :return:
    author: weiwei
    date: 20210120
    """
    if not isinstance(face_id_list, list):
        face_id_list = [face_id_list]
    tmp_vertices = offset_rotmat.dot(objtrm.vertices[objtrm.faces[face_id_list].flatten()].T).T + offset_pos
    tmp_faces = np.array(range(len(tmp_vertices))).reshape(-1, 3)
    return trm.Trimesh(vertices=tmp_vertices, faces=tmp_faces)


def extract_face_center_and_normal(objtrm, face_id_list, offset_pos=np.zeros(3), offset_rotmat=np.eye(3)):
    """
    extract the face center array and the face normal array corresponding to the face id list
    returns a single normal and face center if face_id_list has a single value
    :param objtrm:
    :param face_id_list:
    :param offset_pos:
    :param offset_rotmat:
    :return:
    author: weiwei
    date: 20210120
    """
    return_sgl = False
    if not isinstance(face_id_list, list):
        face_id_list = [face_id_list]
        return_sgl = True
    seed_center_pos_array = offset_rotmat.dot(
        np.mean(objtrm.vertices[objtrm.faces[face_id_list].flatten()], axis=1).reshape(-1, 3).T).T + offset_pos
    seed_normal_array = offset_rotmat.dot(objtrm.face_normals[face_id_list].T).T
    if return_sgl:
        return seed_center_pos_array[0], seed_normal_array[0]
    else:
        return seed_center_pos_array, seed_normal_array


def gen_surface(surface_callback, rng, granularity=.01):
    """
    :param surface_callback:
    :param rng: [[dim0_min, dim0_max], [dim1_min, dim1_max]]
    :return:
    author: weiwei
    date: 20210624
    """

    def _mesh_from_domain_grid(domain_grid, vertices):
        domain_0, domain_1 = domain_grid
        nrow = domain_0.shape[0]
        ncol = domain_0.shape[1]
        faces = np.empty((0, 3))
        for i in range(nrow - 1):
            urgt_pnt0 = np.arange(i * ncol, i * ncol + ncol - 1).T
            urgt_pnt1 = np.arange(i * ncol + 1 + ncol, i * ncol + ncol + ncol).T
            urgt_pnt2 = np.arange(i * ncol + 1, i * ncol + ncol).T
            faces = np.vstack((faces, np.column_stack((urgt_pnt0, urgt_pnt2, urgt_pnt1))))
            blft_pnt0 = np.arange(i * ncol, i * ncol + ncol - 1).T
            blft_pnt1 = np.arange(i * ncol + ncol, i * ncol + ncol + ncol - 1).T
            blft_pnt2 = np.arange(i * ncol + 1 + ncol, i * ncol + ncol + ncol).T
            faces = np.vstack((faces, np.column_stack((blft_pnt0, blft_pnt2, blft_pnt1))))
        return trm.Trimesh(vertices=vertices, faces=faces)

    a_min, a_max = rng[0]
    b_min, b_max = rng[1]
    n_a = round((a_max - a_min) / granularity)
    n_b = round((b_max - b_min) / granularity)
    domain_grid = np.meshgrid(np.linspace(a_min, a_max, n_a, endpoint=True),
                              np.linspace(b_min, b_max, n_b, endpoint=True))
    domain_0, domain_1 = domain_grid
    domain = np.column_stack((domain_0.ravel(), domain_1.ravel()))
    codomain = surface_callback(domain)
    vertices = np.column_stack((domain, codomain))
    return _mesh_from_domain_grid(domain_grid, vertices)


if __name__ == "__main__":
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .2, .3], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    # objcm = gm.WireFrameModel(gen_torus())
    # objcm.set_rgba([1, 0, 0, 1])
    # objcm.attach_to(base)
    # objcm = gm.StaticGeometricModel(gen_axis())
    # objcm.set_rgba([1, 0, 0, 1])
    # objcm.attach_to(base)

    import time

    tic = time.time()
    for i in range(100):
        gen_dumbbell()
    toc = time.time()
    print("mine", toc - tic)
    objcm = gm.GeometricModel(gen_dashstick(lsolid=.005, lspace=.005))
    objcm = gm.GeometricModel(gen_dashtorus(portion=1))
    objcm.set_rgba([1, 0, 0, 1])
    objcm.attach_to(base)
    objcm = gm.GeometricModel(gen_stick())
    objcm.set_rgba([1, 0, 0, 1])
    objcm.set_pos(np.array([0, .01, 0]))
    objcm.attach_to(base)
    objcm = gm.GeometricModel(gen_dasharrow())
    objcm.set_rgba([1, 0, 0, 1])
    objcm.set_pos(np.array([0, -.01, 0]))
    objcm.attach_to(base)
    base.run()
