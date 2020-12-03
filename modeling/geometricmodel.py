import os
import copy
import basis.dataadapter as da
import basis.trimesh as trimesh
import basis.trimeshgenerator as trihelper
import modeling.collisionmodelcollection as cmc
import numpy as np
# import open3d as o3d
from panda3d.core import NodePath, LineSegs, GeomNode, TransparencyAttrib, RenderModeAttrib
from visualization.panda.world import ShowBase


class StaticGeometricModel(object):
    """
    load an object as a static geometric model -> changing pos, rot, color, etc. are not allowed
    there is no extra elements for this model, thus is much faster
    author: weiwei
    date: 20190312
    """

    def __init__(self, objinit=None, btransparency=True, name="defaultname"):
        """
        :param objinit: path type defined by os.path or trimesh or nodepath
        :param btransparency
        :param name
        """
        if isinstance(objinit, StaticGeometricModel):
            self._objpath = copy.deepcopy(objinit.objpath)
            self._trimesh = copy.deepcopy(objinit.trimesh)
            self._pdnp = copy.deepcopy(objinit.pdnp)
            self._name = copy.deepcopy(objinit.name)
            self._localframe = copy.deepcopy(objinit.localframe)
        else:
            if isinstance(objinit, str):
                self._objpath = objinit
                self._trimesh = trimesh.load_mesh(self._objpath)
                self._pdnp = da.trimesh_to_nodepath(self._trimesh)
                self._name = os.path.splitext(os.path.basename(self._objpath))[0]
            elif isinstance(objinit, trimesh.Trimesh):
                self._objpath = None
                self._trimesh = objinit
                self._pdnp = da.trimesh_to_nodepath(objinit)
                self._name = name
            # elif isinstance(objinit, o3d.geometry.TriangleMesh):
            #     self._objpath = None
            #     self._trimesh = trimesh.Trimesh(vertices=objinit.vertices, faces=objinit.triangles,
            #                                      face_normals=objinit.triangle_normals)
            #     self._pdnp = da.trimesh_to_nodepath(self._trimesh)
            #     self._name = name
            # elif isinstance(objinit, o3d.geometry.PointCloud):
            #     self._objpath = None
            #     self._trimesh = trimesh.Trimesh(np.asarray(objinit.points))
            #     self._pdnp = da.nodepath_from_points(self._trimesh.vertices)
            #     self._name = name
            elif isinstance(objinit, np.ndarray):
                self._objpath = None
                self._trimesh = trimesh.Trimesh(objinit)
                self._pdnp = da.nodepath_from_points(self._trimesh.vertices)
                self._name = name
            elif isinstance(objinit, NodePath):
                self._objpath = None
                self._trimesh = None
                self._pdnp = objinit
                self._name = name
            else:
                self._objpath = None
                self._trimesh = None
                self._pdnp = NodePath(name)
                self._name = name
            if btransparency:
                self._pdnp.setTransparency(TransparencyAttrib.MDual)
            self._localframe = None

    @property
    def name(self):
        # read-only property
        return self._name

    @property
    def objpath(self):
        # read-only property
        return self._objpath

    @property
    def pdnp(self):
        # read-only property
        return self._pdnp

    @property
    def trimesh(self):
        # read-only property
        if self._trimesh is None:
            raise ValueError("Only applicable to models with a trimesh!")
        return self._trimesh

    @property
    def localframe(self):
        # read-only property
        return self._localframe

    @property
    def volume(self):
        # read-only property
        if self._trimesh is None:
            raise ValueError("Only applicable to models with a trimesh!")
        return self._trimesh.volume

    def sample_surface(self, radius=0.02, nsample=None):
        """
        :param raidus:
        :return:
        author: weiwei
        date: 20191228
        """
        if self._trimesh is None:
            raise ValueError("Only applicable to models with a trimesh!")
        # do transformation first
        tmptrimesh = self.trimesh.copy()
        tmptrimesh.apply_transform(self.get_homomat())
        if nsample is None:
            nsample = int(round(tmptrimesh.area / ((radius * 0.3) ** 2)))
        samples, faceids = tmptrimesh.sample(nsample, toggle_faceid=True)
        return samples, faceids

    def set_color(self, rgba):
        self._pdnp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def get_color(self):
        return da.pdv4_to_npv4(self._pdnp.getColor())  # panda3d.core.LColor -> LBase4F

    def clear_color(self):
        self._pdnp.clearColor()

    def attach_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._pdnp.reparentTo(obj.render)
        elif isinstance(obj, StaticGeometricModel):
            self._pdnp.reparentTo(obj.pdnp)
        elif isinstance(obj, cmc.CollisionModelCollection):
            obj.addcm(self)
        else:
            print("Must be modeling.StaticGeometricModel, GeometricModel, CollisionModel, or CollisionModelCollection!")

    def remove(self):
        self._pdnp.removeNode()

    def detach(self):
        """
        unshow the object without removing it from memory
        """
        self._pdnp.detachNode()

    def show_localframe(self):
        self._localframe = gen_frame()
        self._localframe.attach_to(self)

    def unshow_localframe(self):
        if self._localframe is not None:
            self._localframe.removeNode()
            self._localframe = None

    def copy(self):
        return StaticGeometricModel(self)


class GeometricModel(StaticGeometricModel):
    """
    load an object as a geometric model
    there is no extra elements for this model, thus is much faster
    author: weiwei
    date: 20190312
    """

    def __init__(self, objinit=None, btransparency=True, name="defaultname"):
        """
        :param objinit: path type defined by os.path or trimesh or nodepath
        """
        if isinstance(objinit, GeometricModel):
            self._objpath = copy.deepcopy(objinit.objpath)
            self._trimesh = copy.deepcopy(objinit.trimesh)
            self._pdnp = copy.deepcopy(objinit.pdnp)
            self._name = copy.deepcopy(objinit.name)
            self._localframe = copy.deepcopy(objinit.localframe)
        else:
            super().__init__(objinit=objinit, btransparency=btransparency, name=name)

    def set_pos(self, npvec3):
        self._pdnp.setPos(npvec3[0], npvec3[1], npvec3[2])

    def get_pos(self):
        return da.pdv3_to_npv3(self._pdnp.getPos())

    def set_rotmat(self, npmat3):
        pdv3 = self._pdnp.getPos()
        pdmat3 = da.npmat3_to_pdmat3(npmat3)
        pdmat4 = da.Mat4(pdmat3, pdv3)
        self._pdnp.setMat(pdmat4)

    def get_rotmat(self):
        pdmat4 = self._pdnp.getMat()
        return da.pdmat4_to_npv3mat3(pdmat4)[1]

    def set_homomat(self, npmat4):
        self._pdnp.setMat(da.npmat4_to_pdmat4(npmat4))

    def get_homomat(self):
        pdmat4 = self._pdnp.getMat()
        return da.pdmat4_to_npmat4(pdmat4)

    def set_rpy(self, roll, pitch, yaw):
        """
        set the pose of the object using rpy
        :param roll: radian
        :param pitch: radian
        :param yaw: radian
        :return:
        author: weiwei
        date: 20190513
        """
        currentmat = self._pdnp.getMat()
        currentmatnp = da.pdmat4_to_npmat4(currentmat)
        newmatnp = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self._pdnp.setMat(da.npv3mat3_to_pdmat4(newmatnp, currentmatnp[:, 3]))

    def get_rpy(self):
        """
        get the pose of the object using rpy
        :return: [r, p, y] in radian
        author: weiwei
        date: 20190513
        """
        currentmat = self._pdnp.getMat()
        currentmatnp = da.pdmat4_to_npmat4(currentmat)
        rpy = rm.rotmat_to_euler(currentmatnp[:3, :3], axes="sxyz")
        return np.array([rpy[0], rpy[1], rpy[2]])

    def set_transparency(self, attribute):
        return self._pdnp.setTransparency(attribute)

    def copy(self):
        return GeometricModel(self)


## primitives are stationarygeometric model, once defined, they cannot be changed
# TODO: further decouple from Panda trimesh->staticgeometricmodel
def gen_linesegs(linesegs, thickness=0.001, rgba=np.array([0, 0, 0, 1])):
    """
    gen linsegs -- non-continuous segs are allowed
    :param linesegs: [[pnt0, pn1], [pnt0, pnt1], ...], pnti 1x3 nparray, defined in local 0 frame
    :param rgba:
    :param thickness:
    :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
    :return: a geomtric model
    author: weiwei
    date: 20161216, 20201116
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * 1000.0)
    for p0p1tuple in linesegs:
        ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        ls.moveTo(p0p1tuple[0][0], p0p1tuple[0][1], p0p1tuple[0][2])
        ls.drawTo(p0p1tuple[1][0], p0p1tuple[1][1], p0p1tuple[1][2])
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


def gen_linesegs(verts, thickness=0.005, rgba=np.array([0, 0, 0, 1])):
    """
    gen continuous linsegs
    :param verts: nx3 list, each nearby pair will be used to draw one segment, defined in a local 0 frame
    :param rgba:
    :param thickness:
    :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
    :return: a geomtric model
    author: weiwei
    date: 20161216
    """
    segs = LineSegs()
    segs.setThickness(thickness * 1000.0)
    segs.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    for i in range(len(verts) - 1):
        tmpstartvert = verts[i]
        tmpendvert = verts[i + 1]
        segs.moveTo(tmpstartvert[0], tmpstartvert[1], tmpstartvert[2])
        segs.drawTo(tmpendvert[0], tmpendvert[1], tmpendvert[2])
    lsnp = NodePath('linesegs')
    lsnp.attachNewNode(segs.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=[1, 0, 0, 1]):
    """
    :param pos:
    :param radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_trm = trihelper.gen_sphere(pos, radius)
    sphere_nodepath = da.trimesh_to_nodepath(sphere_trm)
    sphere_nodepath.setTransparency(TransparencyAttrib.MDual)
    sphere_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    sphere_sgm = StaticGeometricModel(sphere_nodepath)
    return sphere_sgm


def gen_ellipsoid(pos=np.array([0, 0, 0]), axmat=np.eye(3), rgba=[1, 1, 0, .3]):
    """
    :param pos:
    :param axmat: 3x3 mat, each column is an axis of the ellipse
    :param rgba:
    :return:
    author: weiwei
    date: 20200701osaka
    """
    ellipsoid_trm = trihelper.gen_ellipsoid(pos=pos, axmat=axmat)
    ellipsoid_nodepath = da.trimesh_to_nodepath(ellipsoid_trm)
    ellipsoid_nodepath.setTransparency(TransparencyAttrib.MDual)
    ellipsoid_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    ellipsoid_sgm = StaticGeometricModel(ellipsoid_nodepath)
    return ellipsoid_sgm


def gen_stick(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, type="rect",
              rgba=[1, 0, 0, 1]):
    """
    :param spos:
    :param epos:
    :param thickness:
    :param type: rect or round
    :param rgba:
    :return:
    author: weiwei
    date: 20191229osaka
    """
    stick_trm = trihelper.gen_stick(spos=spos, epos=epos, thickness=thickness, type=type)
    stick_nodepath = da.trimesh_to_nodepath(stick_trm)
    stick_nodepath.setTransparency(TransparencyAttrib.MDual)
    stick_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    stick_sgm = StaticGeometricModel(stick_nodepath)
    return stick_sgm


def gen_box(extent=np.array([1, 1, 1]), homomat=np.eye(4), rgba=[1, 0, 0, 1]):
    """
    :param extent:
    :param homomat:
    :return:
    author: weiwei
    date: 20191229osaka
    """
    box_trm = trihelper.gen_box(extent=extent, homomat=homomat)
    box_sgm = StaticGeometricModel(box_trm)
    box_sgm.set_color(rgba=rgba)
    return box_sgm


def gen_dumbbell(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, rgba=[1, 0, 0, 1]):
    """
    :param spos:
    :param epos:
    :param thickness:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    dumbbell_trm = trihelper.gen_dumbbell(spos=spos, epos=epos, thickness=thickness)
    dumbbell_sgm = StaticGeometricModel(dumbbell_trm)
    dumbbell_sgm.set_color(rgba=rgba)
    return dumbbell_sgm


def gen_arrow(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, rgba=[1, 0, 0, 1],
              type="rect"):
    """
    :param spos:
    :param epos:
    :param thickness:
    :param rgba:
    :return:
    author: weiwei
    date: 20200115osaka
    """
    arrow_trm = trihelper.gen_arrow(spos=spos, epos=epos, thickness=thickness, sticktype=type)
    arrow_sgm = StaticGeometricModel(arrow_trm)
    arrow_trm.set_color(rgba=rgba)
    return arrow_sgm


def gen_dasharrow(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, lsolid=None, lspace=None,
                  rgba=[1, 0, 0, 1], type="rect"):
    """
    :param spos:
    :param epos:
    :param thickness:
    :param lsolid: length of the solid section, 1*thickness by default
    :param lspace: length of the empty section, 1.5*thickness by default
    :param rgba:
    :return:
    author: weiwei
    date: 20200625osaka
    """
    dasharrow_trm = trihelper.gen_dasharrow(spos=spos, epos=epos, lsolid=lsolid, lspace=lspace, thickness=thickness,
                                            sticktype=type)
    dasharrow_sgm = StaticGeometricModel(dasharrow_trm)
    dasharrow_sgm.set_color(rgba=rgba)
    return dasharrow_sgm


def gen_frame(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, rgbmatrix=None, alpha=None,
              plotname="frame"):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param length:
    :param thickness:
    :param rgbmatrix: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    endx = pos + rotmat[:, 0] * length
    endy = pos + rotmat[:, 1] * length
    endz = pos + rotmat[:, 2] * length
    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # TODO 20201202 change it to StaticGeometricModelCollection
    frame_nodepath = NodePath(plotname)
    arrowx_trm = trihelper.gen_arrow(spos=pos, epos=endx, thickness=thickness)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trihelper.gen_arrow(spos=pos, epos=endy, thickness=thickness)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = trihelper.gen_arrow(spos=pos, epos=endz, thickness=thickness)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_mycframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, alpha=None, plotname="mycframe"):
    """
    gen an axis for attaching, use magne for x, yellow for y, cyan for z
    :param pos:
    :param rotmat:
    :param length:
    :param thickness:
    :param rgbmatrix: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    rgbmatrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]]).T
    return gen_frame(pos=pos, rotmat=rotmat, length=length, thickness=thickness, rgbmatrix=rgbmatrix, alpha=alpha,
                     plotname=plotname)


def gen_dashframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, lsolid=None, lspace=None,
                  rgbmatrix=None, alpha=None, plotname="dashframe"):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param length:
    :param thickness:
    :param lsolid: length of the solid section, 1*thickness by default
    :param lspace: length of the empty section, 1.5*thickness by default
    :param rgbmatrix: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20200630osaka
    """
    endx = pos + rotmat[:, 0] * length
    endy = pos + rotmat[:, 1] * length
    endz = pos + rotmat[:, 2] * length
    if rgbmatrix is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgbmatrix[:, 0]
        rgby = rgbmatrix[:, 1]
        rgbz = rgbmatrix[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # TODO 20201202 change it to StaticGeometricModelCollection
    frame_nodepath = NodePath(plotname)
    arrowx_trm = trihelper.gen_dasharrow(spos=pos, epos=endx, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trihelper.gen_dasharrow(spos=pos, epos=endy, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = trihelper.gen_dasharrow(spos=pos, epos=endz, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_torus(axis=np.array([1, 0, 0]), portion=.5, center=np.array([0, 0, 0]), radius=.005, thickness=.0015,
              rgba=[1, 0, 0, 1]):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    torus_trm = trihelper.gen_torus(axis=axis, portion=portion, center=center, radius=radius, thickness=thickness)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_color(rgba=rgba)
    return torus_sgm


def gen_circarrow(axis=np.array([1, 0, 0]), portion=.5, center=np.array([0, 0, 0]), radius=.05, thickness=.005,
                  rgba=[1, 0, 0, 1]):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    circarrow_trm = trihelper.gen_circarrow(axis=axis, portion=portion, center=center, radius=radius,
                                            thickness=thickness)
    circarrow_sgm = StaticGeometricModel(circarrow_trm)
    circarrow_sgm.set_color(rgba=rgba)
    return circarrow_sgm


def gen_pointcloud(points, rgbas=[0, 0, 0, .7], pntsize=5):
    """
    do not use this raw function directly
    use environment.collisionmodel to call it
    gen objmnp
    :param points: nx3 list
    :param rgbas: None; Specify for each point; Specify a unified color
    :return:
    """
    pointcloud_nodepath = da.nodepath_from_points(points, rgbas)
    pointcloud_nodepath.setRenderMode(RenderModeAttrib.MPoint, pntsize)
    pointcloud_sgm = StaticGeometricModel(pointcloud_nodepath)
    return pointcloud_sgm


def gen_surface(verts, faces, rgba=[1, 0, 0, 1]):
    """
    TODO 20201202: replace pandanode with trimesh
    :param verts: np.array([[v00, v01, v02], [v10, v11, v12], ...]
    :param faces: np.array([[ti00, ti01, ti02], [ti10, ti11, ti12], ...]
    :param color: rgba
    :return:
    author: weiwei
    date: 20171219
    """
    # gen vert normals
    vertnormals = np.zeros((len(verts), 3))
    for fc in faces:
        vert0 = verts[fc[0], :]
        vert1 = verts[fc[1], :]
        vert2 = verts[fc[2], :]
        facenormal = np.cross(vert2 - vert1, vert0 - vert1)
        vertnormals[fc[0], :] = vertnormals[fc[0]] + facenormal
        vertnormals[fc[1], :] = vertnormals[fc[1]] + facenormal
        vertnormals[fc[2], :] = vertnormals[fc[2]] + facenormal
    for i in range(0, len(vertnormals)):
        vertnormals[i, :] = vertnormals[i, :] / np.linalg.norm(vertnormals[i, :])
    geom = da.pandageom_from_vvnf(verts, vertnormals, faces)
    node = GeomNode('surface')
    node.addGeom(geom)
    surface_nodepath = NodePath('surface')
    surface_nodepath.attachNewNode(node)
    surface_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    surface_nodepath.setTransparency(TransparencyAttrib.MDual)
    surface_nodepath.setTwoSided(True)
    surface_sgm = StaticGeometricModel(surface_nodepath)
    return surface_sgm


def gen_polygon(verts, thickness=0.002, rgba=[0, 0, 0, .7]):
    """
    gen objmnp
    :param objpath:
    :return:a
    author: weiwei
    date: 20201115
    """
    segs = LineSegs()
    segs.setThickness(thickness)
    segs.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    for i in range(len(verts) - 1):
        segs.moveTo(verts[i][0], verts[i][1], verts[i][2])
        segs.drawTo(verts[i + 1][0], verts[i + 1][1], verts[i + 1][2])
    polygon_nodepath = NodePath('polygons')
    polygon_nodepath.attachNewNode(segs.create())
    polygon_nodepath.setTransparency(TransparencyAttrib.MDual)
    polygon_sgm = StaticGeometricModel(polygon_nodepath)
    return polygon_sgm


if __name__ == "__main__":
    import os
    import math
    import numpy as np
    import basis.robotmath as rm
    import visualization.panda.world as wd

    base = wd.World(camp=[.1, .1, .1], lookatpos=[0, 0, 0])
    this_dir, this_filename = os.path.split(__file__)
    objpath = os.path.join(this_dir, "objects", "bunnysim.stl")
    bunnygm = GeometricModel(objpath)
    bunnygm.set_color([0.7, 0.7, 0.0, 1.0])
    bunnygm.attach_to(base)
    bunnygm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnygm.set_rotmat(rotmat)

    bunnygm1 = GeometricModel(objpath)
    bunnygm1.set_color([0.7, 0, 0.7, 1.0])
    bunnygm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnygm1.set_pos(np.array([0, .01, 0]))
    bunnygm1.set_rotmat(rotmat)

    bunnygm2 = GeometricModel(objpath)
    bunnygm2.set_color([0, 0.7, 0.7, 1.0])
    bunnygm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnygm1.set_pos(np.array([0, .2, 0]))
    bunnygm1.set_rotmat(rotmat)

    bunnygmpoints, _ = bunnygm.sample_surface()
    bunnygm1points, _ = bunnygm1.sample_surface()
    bunnygm2points, _ = bunnygm2.sample_surface()
    bpgm = GeometricModel(bunnygmpoints)
    bpgm1 = GeometricModel(bunnygm1points)
    bpgm2 = GeometricModel(bunnygm2points)
    bpgm.attach_to(base)
    bpgm1.attach_to(base)
    bpgm2.attach_to(base)

    lsgm = gen_linesegs(
        [np.array([.1, 0, .01]), np.array([.01, 0, .01]), np.array([.1, 0, .1]), np.array([.1, 0, .01])])
    lsgm.attach_to(base)

    gen_circarrow(radius=.1, portion=.8).attach_to(base)
    gen_dasharrow(spos=np.array([0, 0, 0]), epos=np.array([0, 0, 2])).attach_to(base)
    gen_dashframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3)).attach_to(base)
    axmat = rm.rotmat_from_axangle([1, 1, 1], math.pi / 4)
    gen_frame(rotmat=axmat).attach_to(base)
    axmat[:, 0] = .1 * axmat[:, 0]
    axmat[:, 1] = .07 * axmat[:, 1]
    axmat[:, 2] = .3 * axmat[:, 2]
    gen_ellipsoid(pos=np.array([0, 0, 0]), axmat=axmat).attach_to(base)
    print(rm.unit_vector(np.array([0, 0, 0])))

    base.run()