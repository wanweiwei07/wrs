import numpy as np
import shapely.geometry as shp_geom
import wrs.basis.trimesh.primitives as trm_primit
import wrs.basis.trimesh.geometry as trm_geom
import wrs.basis.trimesh as trm
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm

# declare constants
ARROW_CH_SR = 8  # cone height vs stick radius for arrow
ARROW_CR_SR = 2  # cone bottom radius vs stick radius for arrow


def decorator_check_dimensions(sub_routine):
    def wrapper(spos, epos, *args, **kwargs):
        if spos.ndim != epos.ndim:
            raise ValueError("Different dimensions!")
        return sub_routine(spos, epos, *args, **kwargs)

    return wrapper


def trm_from_vvnf(vertices, vertex_normals, faces):
    return trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)


def gen_box(xyz_lengths=np.array([1, 1, 1]),
            pos=np.zeros(3),
            rotmat=np.eye(3)):
    """
    :param xyz_lengths: x, y, z (origin is 0)
    :param pos: rotation and translation
    :return: a Trimesh object (Primitive)
    author: weiwei
    date: 20191228osaka
    """
    return trm_primit.Box(extents=xyz_lengths, homomat=rm.homomat_from_posrot(pos=pos, rotmat=rotmat))


def gen_frustrum(bottom_xy_lengths=np.array([0.02, 0.02]), top_xy_lengths=np.array([0.04, 0.04]), height=0.01,
                 pos=np.zeros(3), rotmat=np.eye(3)):
    """
    Draw a 3D frustum
    :param bottom_xy_lengths: XYZ lengths of the bottom rectangle
    :param top_xy_lengths: XYZ lengths of the top rectangle
    :param height: Height of the frustum
    :param pos: Position of the frustum center
    :param rotmat: Rotation matrix for the frustum orientation
    :return: A NodePath with the frustum geometry
    """
    vertices = np.array([
        [-bottom_xy_lengths[0] / 2, -bottom_xy_lengths[1] / 2, 0],
        [bottom_xy_lengths[0] / 2, -bottom_xy_lengths[1] / 2, 0],
        [bottom_xy_lengths[0] / 2, bottom_xy_lengths[1] / 2, 0],
        [-bottom_xy_lengths[0] / 2, bottom_xy_lengths[1] / 2, 0],
        [-top_xy_lengths[0] / 2, -top_xy_lengths[1] / 2, height],
        [top_xy_lengths[0] / 2, -top_xy_lengths[1] / 2, height],
        [top_xy_lengths[0] / 2, top_xy_lengths[1] / 2, height],
        [-top_xy_lengths[0] / 2, top_xy_lengths[1] / 2, height]
    ])
    # Define the faces of the frustum
    faces = np.array([
        [0, 2, 1], [2, 0, 3],  # Bottom face
        [4, 5, 6], [6, 7, 4],  # Top face
        [0, 1, 5], [5, 4, 0],  # Side face 1
        [1, 2, 6], [6, 5, 1],  # Side face 2
        [2, 3, 7], [7, 6, 2],  # Side face 3
        [3, 0, 4], [4, 7, 3]  # Side face 4
    ])
    frustum_mesh = trm.Trimesh(vertices=vertices, faces=faces)
    frustum_mesh.apply_transform(rm.homomat_from_posrot(pos, rotmat))
    return frustum_mesh


def gen_stick(spos=np.array([0, 0, 0]),
              epos=np.array([0.1, 0, 0]),
              radius=0.0025,
              type="rect",
              n_sec=8):
    """
    interface to genrectstick/genroundstick
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param radius: 0.005 m by default
    :param type: rect or round
    :param n_sec: how many pie wedges should the cylinder be meshed as
    :return:
    author: weiwei
    date: 20191228osaka
    """
    if type == "rect":
        return gen_rectstick(spos=spos, epos=epos, radius=radius, n_sec=n_sec)
    if type == "round":
        return gen_roundstick(spos=spos, epos=epos, radius=radius, n_sec=n_sec)


def gen_rectstick(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), radius=.0025, n_sec=8):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param radius: 0.005 m by default
    :param n_sec: how many pie wedges should the cylinder be meshed as
    :return: a Trimesh object (Primitive)
    author: weiwei
    date: 20191228osaka
    """
    height = np.linalg.norm(epos - spos)
    if np.allclose(height, 0):
        rotmat = np.eye(3)
    else:
        rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot((spos + epos) / 2.0, rotmat)
    cylinder = trm_primit.Cylinder(height=height, radius=radius, n_sec=n_sec, homomat=homomat)
    return cylinder


def gen_roundstick(spos=np.array([0, 0, 0]),
                   epos=np.array([0.1, 0, 0]),
                   radius=0.005,
                   n_sec=8):
    """
    :param spos:
    :param epos:
    :param radius:
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
    return trm_primit.Capsule(height=height,
                              radius=radius,
                              count=[n_sec, n_sec],
                              homomat=homomat)


@decorator_check_dimensions
def gen_dashstick(spos=np.array([0, 0, 0]),
                  epos=np.array([0.1, 0, 0]),
                  radius=0.0025,
                  len_solid=None,
                  len_interval=None,
                  n_sec=8,
                  type="rect"):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param radius: 0.005 m by default
    :param len_solid: axis_length of the solid section, 1*major_radius if None
    :param len_interval: axis_length of the interval between two solids, 1.5*major_radius if None
    :return:
    author: weiwei
    date: 20191228osaka
    """
    if len_solid is None:
        len_solid = radius * 3.2
    if len_interval is None:
        len_interval = radius * 2.14
    length, direction = rm.unit_vector(epos - spos, toggle_length=True)
    n_solid = round(length / (len_solid + len_interval) + 0.5)
    vertices = np.empty((0, 3))
    faces = np.empty((0, 3))
    for i in range(0, n_solid - 1):
        tmp_spos = spos + (len_solid * direction + len_interval * direction) * i
        tmp_stick = gen_stick(spos=tmp_spos,
                              epos=tmp_spos + len_solid * direction,
                              radius=radius,
                              type=type,
                              n_sec=n_sec)
        tmp_stick_faces = tmp_stick.faces + len(vertices)
        vertices = np.vstack((vertices, tmp_stick.vertices))
        faces = np.vstack((faces, tmp_stick_faces))
    # wrap up the last segment
    tmp_spos = spos + (len_solid * direction + len_interval * direction) * (n_solid - 1)
    tmp_epos = tmp_spos + len_solid * direction
    final_length, _ = rm.unit_vector(tmp_epos - spos, toggle_length=True)
    if final_length > length:
        tmp_epos = epos
    tmp_stick = gen_stick(spos=tmp_spos,
                          epos=tmp_epos,
                          radius=radius,
                          type=type,
                          n_sec=n_sec)
    tmp_stick_faces = tmp_stick.faces + len(vertices)
    vertices = np.vstack((vertices, tmp_stick.vertices))
    faces = np.vstack((faces, tmp_stick_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.02, ico_level=2):
    """
    :param pos: 1x3 nparray
    :param radius: 0.02 m by default
    :param ico_level: levels of icosphere n_sec_major
    :return:
    author: weiwei
    date: 20191228osaka
    """
    return trm_primit.Sphere(radius=radius, center=pos, ico_level=ico_level)


def gen_ellipsoid(pos=np.array([0, 0, 0]), axmat=np.eye(3), subdivisions=5):
    """
    :param pos:
    :param axmat: 3x3 mat, each column is an axis of the ellipse
    :param subdivisions: levels of icosphere n_sec_major
    :return:
    author: weiwei
    date: 20191228osaka
    """
    sphere = trm_primit.Sphere(sphere_radius=1, sphere_center=np.zeros(3), subdivisions=subdivisions)
    vertices = axmat.dot(sphere.vertices.T).T
    vertices = vertices + pos
    return trm.Trimesh(vertices=vertices, faces=sphere.faces)


def gen_dumbbell(spos=np.array([0, 0, 0]),
                 epos=np.array([0.1, 0, 0]),
                 stick_radius=0.005,
                 n_sec=8,
                 sphere_radius=None,
                 sphere_ico_level=2):
    """
    NOTE: return stick+spos_ball+epos_ball also work, but it is a bit slower
    :param sphere_radius:
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param stick_radius: 0.005 m by default
    :param n_sec: how many pie wedges should the cylinder be meshed as
    :param sphere_radius: 1.5*radius if None
    :param sphere_ico_level: levels of icosphere n_sec_major
    :return:
    author: weiwei
    date: 20191228osaka
    """
    if sphere_radius is None:
        sphere_radius = 1.5 * stick_radius
    stick = gen_rectstick(spos=spos, epos=epos, radius=stick_radius, n_sec=n_sec)
    spos_ball = gen_sphere(pos=spos, radius=sphere_radius, ico_level=sphere_ico_level)
    epos_ball = gen_sphere(pos=epos, radius=sphere_radius, ico_level=sphere_ico_level)
    vertices = np.vstack((stick.vertices, spos_ball.vertices, epos_ball.vertices))
    sposballfaces = spos_ball.faces + len(stick.vertices)
    endballfaces = epos_ball.faces + len(spos_ball.vertices) + len(stick.vertices)
    faces = np.vstack((stick.faces, sposballfaces, endballfaces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_cone(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), bottom_radius=0.0025, n_sec=8):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param bottom_radius: 0.0025 m by default
    :param n_sec: how many pie wedges should the cylinder be meshed as
    :return:
    author: weiwei
    date: 20191228osaka, 20230812toyonaka
    """
    height = np.linalg.norm(spos - epos)
    rotmat = rm.rotmat_between_vectors(np.array([0, 0, 1]), epos - spos)
    homomat = rm.homomat_from_posrot(spos, rotmat)
    return trm_primit.Cone(height=height, radius=bottom_radius, n_sec=n_sec, homomat=homomat)


def gen_arrow(spos=np.array([0, 0, 0]), epos=np.array([0.1, 0, 0]), stick_radius=0.0025, n_sec=8, stick_type="rect"):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param stick_radius: 0.005 m by default
    :param n_sec: how many pie wedges should the cylinder be meshed as
    :param stick_type: The shape at the end_type of the arrow stick, round or rect
    :param major_radius:
    :return:
    author: weiwei
    date: 20191228osaka
    """
    direction = rm.unit_vector(epos - spos)
    if np.linalg.norm(spos - epos) > stick_radius * ARROW_CH_SR:
        stick = gen_stick(spos=spos,
                          epos=epos - direction * stick_radius * ARROW_CH_SR,
                          radius=stick_radius,
                          type=stick_type,
                          n_sec=n_sec)
        cap = gen_cone(spos=epos - direction * stick_radius * ARROW_CH_SR,
                       epos=epos,
                       bottom_radius=stick_radius * ARROW_CR_SR,
                       n_sec=n_sec)
        vertices = np.vstack((stick.vertices, cap.vertices))
        capfaces = cap.faces + len(stick.vertices)
        faces = np.vstack((stick.faces, capfaces))
    else:
        cap = gen_cone(spos=epos - direction * stick_radius * ARROW_CH_SR,
                       epos=epos,
                       bottom_radius=stick_radius * ARROW_CR_SR,
                       n_sec=n_sec)
        vertices = cap.vertices
        faces = cap.faces
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_dashed_arrow(spos=np.array([0, 0, 0]),
                     epos=np.array([0.1, 0, 0]),
                     stick_radius=0.0025,
                     len_solid=None,
                     len_interval=None,
                     n_sec=8,
                     stick_type="rect"):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param stick_radius: 0.005 m by default
    :param len_solid: axis_length of the solid section, 1*major_radius if None
    :param len_interval: axis_length of the empty section, 1.5*major_radius if None
    :param n_sec:
    :param stick_type:
    :return:
    author: weiwei
    date: 20191228osaka, 20230812toyonaka
    """
    length, direction = rm.unit_vector(epos - spos, toggle_length=True)
    cap = gen_cone(spos=epos - direction * stick_radius * ARROW_CH_SR,
                   epos=epos,
                   bottom_radius=stick_radius * ARROW_CR_SR,
                   n_sec=n_sec)
    dash_stick = gen_dashstick(spos=epos - direction * stick_radius * ARROW_CH_SR,
                               epos=spos,  # make sure the section near the arrow head is even and solid
                               radius=stick_radius,
                               len_solid=len_solid,
                               len_interval=len_interval,
                               n_sec=n_sec,
                               type=stick_type)
    tmp_stick_faces = dash_stick.faces + len(cap.vertices)
    vertices = np.vstack((cap.vertices, dash_stick.vertices))
    faces = np.vstack((cap.faces, tmp_stick_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_frame(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=0.1, stick_radius=0.0025):
    """
    :param spos: 1x3 nparray
    :param epos: 1x3 nparray
    :param stick_radius: 0.005 m by default
    :return:
    author: weiwei
    date: 20191228osaka
    """
    direction_x = rotmat[:, 0]
    direction_y = rotmat[:, 1]
    direction_z = rotmat[:, 2]
    # x
    end_x = direction_x * length
    stick_x = gen_stick(spos=pos, epos=end_x, radius=stick_radius)
    cone_x = gen_cone(spos=end_x,
                      epos=end_x + direction_x * stick_radius * ARROW_CH_SR,
                      bottom_radius=stick_radius * ARROW_CR_SR)
    # y
    end_y = direction_y * length
    stick_y = gen_stick(spos=pos, epos=end_y, radius=stick_radius)
    cone_y = gen_cone(spos=end_y,
                      epos=end_y + direction_y * stick_radius * ARROW_CH_SR,
                      bottom_radius=stick_radius * ARROW_CR_SR)
    # z
    end_z = direction_z * length
    stick_z = gen_stick(spos=pos, epos=end_z, radius=stick_radius)
    cone_z = gen_cone(spos=end_z,
                      epos=end_z + direction_z * stick_radius * ARROW_CH_SR,
                      bottom_radius=stick_radius * ARROW_CR_SR)
    vertices = np.vstack(
        (stick_x.vertices, cone_x.vertices, stick_y.vertices, cone_y.vertices, stick_z.vertices, cone_z.vertices))
    cone_x_faces = cone_x.faces + len(stick_x.vertices)
    stick_y_faces = stick_y.faces + len(stick_x.vertices) + len(cone_x.vertices)
    cone_y_faces = cone_y.faces + len(stick_x.vertices) + len(cone_x.vertices) + len(stick_y.vertices)
    stick_z_faces = stick_z.faces + len(stick_x.vertices) + len(cone_x.vertices) + len(stick_y.vertices) + len(
        cone_y.vertices)
    cone_z_faces = cone_z.faces + len(stick_x.vertices) + len(cone_x.vertices) + len(stick_y.vertices) + len(
        cone_y.vertices) + len(stick_z.vertices)
    faces = np.vstack((stick_x.faces, cone_x_faces, stick_y_faces, cone_y_faces, stick_z_faces, cone_z_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=.5,
              center=np.array([0, 0, 0]),
              major_radius=0.1,
              minor_radius=0.0025,
              n_sec_major=24,
              n_sec_minor=8):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param major_radius:
    :param minor_radius:
    :param n_sec_major: number sticks used for approximating a torus
    :param n_sec_minor: # of discretized sectors used to approximate a cylindrical stick
    :return:
    author: weiwei
    date: 20200602, 20230812
    """
    unit_axis = rm.unit_vector(axis)
    if starting_vector is None:
        starting_vector = rm.orthogonal_vector(unit_axis)
    else:
        starting_vector = rm.unit_vector(starting_vector)
    starting_pos = starting_vector * major_radius + center
    major_sec_angle = 2 * np.pi / n_sec_major
    n_div = round(portion * n_sec_major)
    # gen the last sec first
    # gen the remaining torus afterwards
    if n_div > 0:
        prev_pos = center + np.dot(rm.rotmat_from_axangle(unit_axis, (n_div - 1) * major_sec_angle),
                                   starting_vector) * major_radius
        nxt_pos = center + np.dot(rm.rotmat_from_axangle(unit_axis, portion * 2 * np.pi),
                                  starting_vector) * major_radius
        stick = gen_stick(spos=prev_pos, epos=nxt_pos, radius=minor_radius, n_sec=n_sec_minor, type="round")
        vertices = stick.vertices
        faces = stick.faces
        prev_pos = starting_pos
        for i in range(1 * np.sign(n_div), n_div, 1 * np.sign(n_div)):
            nxt_pos = center + np.dot(rm.rotmat_from_axangle(unit_axis, i * major_sec_angle),
                                      starting_vector) * major_radius
            stick = gen_stick(spos=prev_pos, epos=nxt_pos, radius=minor_radius, n_sec=n_sec_minor, type="round")
            stick_faces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stick_faces))
            prev_pos = nxt_pos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        raise ValueError("The n_sec_major value is to small for generating a torus with the given portion." +
                         f" The current portion * n_sec_major value is {portion * n_sec_major}. It should > 1.0.")


def gen_dashtorus(axis=np.array([1, 0, 0]),
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  major_radius=0.1,
                  minor_radius=0.0025,
                  len_solid=None,
                  len_interval=None,
                  n_sec_major=24,
                  n_sec_minor=8):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param major_radius:
    :param minor_radius:
    :param len_solid: axis_length of solid
    :param len_interval: axis_length of space
    :param n_sec_major: number sticks used for approximating a torus
    :param n_sec_minor: # of discretized sectors used to approximate a cylindrical stick
    :return:
    author: weiwei
    date: 20200602, 20230812
    """
    if portion > 1 or portion <= 0:
        raise ValueError("The value of portion must be in (0,1]!")
    if not len_solid:
        len_solid = minor_radius * 3.2
    if not len_interval:
        len_interval = minor_radius * 2.14
    unit_axis = rm.unit_vector(axis)
    starting_vector = rm.orthogonal_vector(unit_axis)
    # make sure n_sec_major is large enough to support one dash section.
    min_n_sec_major = np.ceil(2 * np.pi * major_radius / len_solid)
    # print(min_n_sec_major)
    if n_sec_major < min_n_sec_major:
        n_sec_major = min_n_sec_major
    n_sec_major_portion = np.floor(portion * 2 * np.pi * major_radius / (len_solid + len_interval)).astype(int)
    vertices = np.empty((0, 3))
    faces = np.empty((0, 3))
    for i in range(0, n_sec_major_portion):
        tmp_vec = rm.rotmat_from_axangle(axis, 2 * np.pi * portion / n_sec_major_portion * i).dot(starting_vector)
        torus_sec = gen_torus(axis=axis,
                              starting_vector=tmp_vec,
                              portion=portion / n_sec_major_portion * len_solid / (len_solid + len_interval),
                              center=center,
                              major_radius=major_radius,
                              minor_radius=minor_radius,
                              n_sec_major=n_sec_major,
                              n_sec_minor=n_sec_minor)
        torus_sec_faces = torus_sec.faces + len(vertices)
        vertices = np.vstack((vertices, torus_sec.vertices))
        faces = np.vstack((faces, torus_sec_faces))
    return trm.Trimesh(vertices=vertices, faces=faces)


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=0.3,
                  center=np.array([0, 0, 0]),
                  major_radius=0.005,
                  minor_radius=0.00075,
                  n_sec_major=24,
                  n_sec_minor=8,
                  end_type='single'):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param major_radius:
    :param minor_radius:
    :param rgba:
    :param n_sec_major: number sticks used for approximation
    :param end_type: 'single' or 'double'
    :return:
    author: weiwei
    date: 20200602
    """
    unit_axis = rm.unit_vector(axis)
    if starting_vector is None:
        starting_vector = rm.orthogonal_vector(unit_axis)
    else:
        starting_vector = rm.unit_vector(starting_vector)
    starting_pos = starting_vector * major_radius + center
    major_sec_angle = 2 * np.pi / n_sec_major
    n_div = int(portion * n_sec_major + .5)
    # step 1: gen the last arrow first
    # step 2: gen the remaining torus
    if n_div > 0:
        arrow_ticks = round(minor_radius * ARROW_CH_SR / (major_sec_angle * major_radius) + .5)
        prev_pos = center + np.dot(
            rm.rotmat_from_axangle(unit_axis, portion * 2 * np.pi - arrow_ticks * major_sec_angle),
            starting_vector) * major_radius
        nxt_pos = center + np.dot(rm.rotmat_from_axangle(unit_axis, portion * 2 * np.pi),
                                  starting_vector) * major_radius
        arrow = gen_arrow(spos=prev_pos, epos=nxt_pos, stick_radius=minor_radius, n_sec=n_sec_minor, stick_type="round")
        vertices = arrow.vertices
        faces = arrow.faces
        if end_type == 'single':
            id_start = 1
            prev_pos = starting_pos
        elif end_type == 'double':
            id_start = arrow_ticks
            prev_pos = center + np.dot(rm.rotmat_from_axangle(unit_axis, (arrow_ticks) * major_sec_angle),
                                       starting_vector) * major_radius
            nxt_pos = starting_pos
            arrow = gen_arrow(spos=prev_pos, epos=nxt_pos, stick_radius=minor_radius, n_sec=n_sec_minor,
                              stick_type="round")
            arrow_faces = arrow.faces + len(vertices)
            vertices = np.vstack((vertices, arrow.vertices))
            faces = np.vstack((faces, arrow_faces))
        for i in range(id_start, n_div - arrow_ticks + 2, 1):
            if i == n_div - arrow_ticks + 1:
                nxt_pos = center + np.dot(
                    rm.rotmat_from_axangle(unit_axis, portion * 2 * np.pi - arrow_ticks * major_sec_angle),
                    starting_vector) * major_radius
            else:
                nxt_pos = center + np.dot(rm.rotmat_from_axangle(unit_axis, i * major_sec_angle),
                                          starting_vector) * major_radius
            stick = gen_stick(spos=prev_pos, epos=nxt_pos, radius=minor_radius, n_sec=n_sec_minor, type="round")
            stick_faces = stick.faces + len(vertices)
            vertices = np.vstack((vertices, stick.vertices))
            faces = np.vstack((faces, stick_faces))
            prev_pos = nxt_pos
        return trm.Trimesh(vertices=vertices, faces=faces)
    else:
        return trm.Trimesh()


def facet_boundary(trm_mesh, facet, facet_center, facet_normal):
    """
    compute a boundary polygon for facet
    assumptions:
    1. there is only one boundary
    2. the facet is convex
    :param trm_mesh: a datatype defined in trimesh
    :param facet: a data end_type defined in trimesh
    :param facet_center and facet_normal used to compute the transform, see trimesh.geometry.plane_transform
    :return: [1x3 vertices list, 1x2 vertices list, 4x4 homogeneous transformation matrix)]
    author: weiwei
    date: 20161213tsukuba
    """
    shape_face_merged = None
    # use -facet_normal to let the it face downward
    facet_homomat = trm_geom.plane_transform(facet_center, -facet_normal)
    for i, faceidx in enumerate(facet):
        vert0 = trm_mesh.vertices[trm_mesh.faces[faceidx][0]]
        vert1 = trm_mesh.vertices[trm_mesh.faces[faceidx][1]]
        vert2 = trm_mesh.vertices[trm_mesh.faces[faceidx][2]]
        vert0_transformed = rm.transform_points_by_homomat(facet_homomat, vert0)
        vert1_transformed = rm.transform_points_by_homomat(facet_homomat, vert1)
        vert2_transformed = rm.transform_points_by_homomat(facet_homomat, vert2)
        shp_face = shp_geom.Polygon([vert0_transformed[:2], vert1_transformed[:2], vert2_transformed[:2]])
        if shape_face_merged is None:
            shape_face_merged = shp_face
        else:
            shape_face_merged = shape_face_merged.union(shp_face)
    verts_2d = list(shape_face_merged.exterior.coords)
    verts_3d = []
    for vert in verts_2d:
        vert_transformed = rm.transform_points_by_homomat(rm.homomat_inverse(facet_homomat),
                                                          np.array([vert[0], vert[1], 0]))[:3]
        verts_3d.append(vert_transformed)
    return verts_3d, verts_2d, facet_homomat


def facet_as_trm(trm_mesh, facet, offset_pos=np.zeros(3), offset_rotmat=np.eye(3)):
    """
    TODO: return multiple trimesh to make sure a single trimesh is a connected piece
    :param trm_mesh:
    :param facet: one or a list of facets
    :param offset_pos:
    :param offset_rotmat:
    :return:
    author: weiwei
    date: 20210120
    """
    if not isinstance(facet, list):
        facet = [facet]
    vertices = offset_rotmat.dot(trm_mesh.vertices[trm_mesh.faces[facet].flatten()].T).T + offset_pos
    faces = np.array(range(len(vertices))).reshape(-1, 3)
    return trm.Trimesh(vertices=vertices, faces=faces)


def facet_center_and_normal(trm_mesh, facet, offset_pos=np.zeros(3), offset_rotmat=np.eye(3)):
    """
    extract the face center array and the face normal array corresponding to the face id list
    returns a single normal and face center if facet has a single value
    :param trm_mesh:
    :param facet: one or a list of facets
    :param offset_pos:
    :param offset_rotmat:
    :return:
    author: weiwei
    date: 20210120
    """
    return_sgl = False
    if not isinstance(facet, list):
        facet = [facet]
        return_sgl = True
    seed_center_pos_array = offset_rotmat.dot(
        np.mean(trm_mesh.vertices[trm_mesh.faces[facet].flatten()], axis=1).reshape(-1, 3).T).T + offset_pos
    seed_normal_array = offset_rotmat.dot(trm_mesh.face_normals[facet].T).T
    if return_sgl:
        return seed_center_pos_array[0], seed_normal_array[0]
    else:
        return seed_center_pos_array, seed_normal_array


def gen_surface(surface_callback, range, granularity=.01):
    """
    :param surface_callback:
    :param range: [[dim0_min, dim0_max], [dim1_min, dim1_max]]
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

    a_min, a_max = range[0]
    b_min, b_max = range[1]
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
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .2, .3], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    # obj_cmodel = mgm.WireFrameModel(gen_torus())
    # obj_cmodel.set_rgba([1, 0, 0, 1])
    # obj_cmodel.attach_to(base)
    objcm = mgm.StaticGeometricModel(gen_frame())
    objcm.rgba(np.array([1, 0, 0, 1]))
    objcm.attach_to(base)
    # base.run()

    # tic = time.time()
    # for i in range(100):
    #     spe = mgm.GeometricModel(gen_sphere(bottom_radius=.1))
    # toc = time.time()
    # spe.attach_to(base)
    # print("mine", toc - tic)
    # obj_cmodel = mgm.GeometricModel(gen_dashstick(len_solid=.005, len_interval=.005))
    objcm = mgm.GeometricModel(gen_dashtorus(portion=.5))
    objcm.set_rgba([1, 0, 0, 1])
    objcm.attach_to(base)
    base.run()
    objcm = mgm.GeometricModel(gen_stick())
    objcm.set_rgba([1, 0, 0, 1])
    objcm.set_pos(np.array([0, .01, 0]))
    objcm.attach_to(base)
    objcm = mgm.GeometricModel(gen_dashed_arrow())
    objcm.set_rgba([1, 0, 0, 1])
    objcm.set_pos(np.array([0, -.01, 0]))
    objcm.attach_to(base)
    base.run()
