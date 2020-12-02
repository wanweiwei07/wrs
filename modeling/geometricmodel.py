import os
import math
import copy
import modeling.datahelper as dh
import modeling.trimesh as trimesh
import modeling.trimesh.sample as sample
import modeling.trimesh.helper as trihelper
import modeling.collisionmodelcollection as cmc
import numpy as np
import open3d as o3d
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
                self._pdnp = dh.trimesh_to_nodepath(self._trimesh)
                self._name = os.path.splitext(os.path.basename(self._objpath))[0]
            elif isinstance(objinit, trimesh.Trimesh):
                self._objpath = None
                self._trimesh = objinit
                self._pdnp = dh.trimesh_to_nodepath(objinit)
                self._name = name
            elif isinstance(objinit, o3d.geometry.TriangleMesh):
                self._objpath = None
                self._trimesh = trimesh.Trimesh(vertices=objinit.vertices, faces=objinit.triangles,
                                                 face_normals=objinit.triangle_normals)
                self._pdnp = dh.trimesh_to_nodepath(self._trimesh)
                self._name = name
            elif isinstance(objinit, o3d.geometry.PointCloud):
                self._objpath = None
                self._trimesh = trimesh.Trimesh(np.asarray(objinit.points))
                self._pdnp = dh.nodepath_from_points(self._trimesh.vertices)
                self._name = name
            elif isinstance(objinit, np.ndarray):
                self._objpath = None
                self._trimesh = trimesh.Trimesh(objinit)
                self._pdnp = dh.nodepath_from_points(self._trimesh.vertices)
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
        tmptrimesh.apply_transform(self.gethomomat())
        if nsample is None:
            nsample = int(math.ceil(tmptrimesh.area / ((radius * 0.3) ** 2)))
        samples, faceids = sample.sample_surface_even_withfaceid(tmptrimesh, nsample)
        return samples, faceids

    def reparent_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._pdnp.reparentTo(obj.render)
        elif isinstance(obj, StaticGeometricModel):
            self._pdnp.reparentTo(obj.pdnp)
        elif isinstance(obj, cmc.CollisionModelCollection):
            obj.addcm(self)
        else:
            print("Argument 1 must be modeling.StaticGeometricModel, GeometricModel, CollisionModel, or CollisionModelCollection")

    def remove(self):
        self._pdnp.removeNode()

    def detach(self):
        """
        unshow the object without removing it from memory
        """
        self._pdnp.detachNode()

    def showlocalframe(self):
        self._localframe = genframe()
        self._localframe.reparent_to(self.pdnp)

    def unshowlocalframe(self):
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

    def setpos(self, npvec3):
        self._pdnp.setPos(npvec3[0], npvec3[1], npvec3[2])

    def getpos(self):
        return dh.pdv3_to_npv3(self._pdnp.getPos())

    def setrotmat(self, npmat3):
        pdv3 = self._pdnp.getPos()
        pdmat3 = dh.npmat3_to_pdmat3(npmat3)
        pdmat4 = dh.Mat4(pdmat3, pdv3)
        self._pdnp.setMat(pdmat4)

    def getrotmat(self):
        pdmat4 = self._pdnp.getMat()
        return dh.pdmat4_to_npv3mat3(pdmat4)[1]

    def sethomomat(self, npmat4):
        self._pdnp.setMat(dh.npmat4_to_pdmat4(npmat4))

    def gethomomat(self):
        pdmat4 = self._pdnp.getMat()
        return dh.pdmat4_to_npmat4(pdmat4)

    def setrpy(self, roll, pitch, yaw):
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
        currentmatnp = dh.pdmat4_to_npmat4(currentmat)
        newmatnp = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self._pdnp.setMat(dh.npv3mat3_to_pdmat4(newmatnp, currentmatnp[:, 3]))

    def getrpy(self):
        """
        get the pose of the object using rpy
        :return: [r, p, y] in radian
        author: weiwei
        date: 20190513
        """
        currentmat = self._pdnp.getMat()
        currentmatnp = dh.pdmat4_to_npmat4(currentmat)
        rpy = rm.rotmat_to_euler(currentmatnp[:3, :3], axes="sxyz")
        return np.array([rpy[0], rpy[1], rpy[2]])

    def settransparency(self, attribute):
        return self._pdnp.setTransparency(attribute)

    def setcolor(self, rgba):
        self._pdnp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def getcolor(self):
        return dh.pdv4_to_npv4(self._pdnp.getColor()) # panda3d.core.LColor -> LBase4F

    def clearcolor(self):
        self._pdnp.clearColor()

    def copy(self):
        return GeometricModel(self)


## primitives are stationarygeometric model, once defined, they cannot be changed
# TODO: further decouple from Panda trimesh->staticgeometricmodel
def genlinesegs(linesegs, rgba=np.array([0, 0, 0, 1]), thickness=0.001):
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
    ls.setThickness(thickness*1000.0)
    for p0p1tuple in linesegs:
        ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
        ls.moveTo(p0p1tuple[0][0], p0p1tuple[0][1], p0p1tuple[0][2])
        ls.drawTo(p0p1tuple[1][0], p0p1tuple[1][1], p0p1tuple[1][2])
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


def genlinesegs(verts, rgba=np.array([0, 0, 0, 1]), thickness=0.005):
    """
    gen continuous linsegs
    :param verts: [pnt0, pnt1, pnt2, ...] each nearby pair will be used to draw one segment, defined in a local 0 frame
    :param rgba:
    :param thickness:
    :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
    :return: a geomtric model
    author: weiwei
    date: 20161216
    """
    segs = LineSegs()
    segs.setThickness(thickness*1000.0)
    segs.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    for i in range(len(verts) - 1):
        tmpstartvert = verts[i]
        tmpendvert = verts[i+1]
        segs.moveTo(tmpstartvert[0], tmpstartvert[1], tmpstartvert[2])
        segs.drawTo(tmpendvert[0], tmpendvert[1], tmpendvert[2])
    lsnp = NodePath('linesegs')
    lsnp.attachNewNode(segs.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


def gensphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=np.array([1, 0, 0, 1])):
    """
    :param pos:
    :param radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_trm = trihelper.gensphere(pos, radius)
    sphere_nodepath = dh.trimesh_to_nodepath(sphere_trm)
    sphere_nodepath.setTransparency(TransparencyAttrib.MDual)
    sphere_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    sphere_sgm = StaticGeometricModel(sphere_nodepath)
    return sphere_sgm


def genellipsoid(pos=np.array([0, 0, 0]), axmat=np.eye(3), rgba=np.array([1, 1, 0, .3])):
    """
    :param pos:
    :param axmat: 3x3 mat, each column is an axis of the ellipse
    :param rgba:
    :return:
    author: weiwei
    date: 20200701osaka
    """
    ellipsoid_trm = trihelper.genellipsoid(pos=pos, axmat=axmat)
    ellipsoid_nodepath = dh.trimesh_to_nodepath(ellipsoid_trm)
    ellipsoid_nodepath.setTransparency(TransparencyAttrib.MDual)
    ellipsoid_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    ellipsoid_sgm = StaticGeometricModel(ellipsoid_nodepath)
    return ellipsoid_sgm


def genstick(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, type="rect",
             rgba=np.array([1, 0, 0, 1])):
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
    stick_trm = trihelper.genstick(spos=spos, epos=epos, thickness=thickness, type=type)
    stick_nodepath= dh.trimesh_to_nodepath(stick_trm)
    stick_nodepath.setTransparency(TransparencyAttrib.MDual)
    stick_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    stick_sgm = StaticGeometricModel(stick_nodepath)
    return stick_sgm


def genbox(extent=np.array([1, 1, 1]), homomat=np.eye(4), rgba=np.array([1, 0, 0, 1])):
    """
    :param extent:
    :param homomat:
    :return:
    author: weiwei
    date: 20191229osaka
    """
    box_trm = trihelper.genbox(extent=extent, homomat=homomat)
    box_nodepath = dh.trimesh_to_nodepath(box_trm)
    box_nodepath.setTransparency(TransparencyAttrib.MDual)
    box_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    box_sgm = StaticGeometricModel(box_nodepath)
    return box_sgm


def gendumbbell(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, rgba=np.array([1, 0, 0, 1])):
    """
    :param spos:
    :param epos:
    :param thickness:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    dumbbell_trm = trihelper.gendumbbell(spos=spos, epos=epos, thickness=thickness)
    dumbbell_nodepath = dh.trimesh_to_nodepath(dumbbell_trm)
    dumbbell_nodepath.setTransparency(TransparencyAttrib.MDual)
    dumbbell_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    dumbbell_sgm = StaticGeometricModel(dumbbell_nodepath)
    return dumbbell_sgm


def genarrow(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, rgba=np.array([1, 0, 0, 1]),
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
    arrow_trm = trihelper.genarrow(spos=spos, epos=epos, thickness=thickness, sticktype=type)
    arrow_nodepath = dh.trimesh_to_nodepath(arrow_trm)
    arrow_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrow_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    arrow_sgm = StaticGeometricModel(arrow_nodepath)
    return arrow_sgm


def gendasharrow(spos=np.array([0, 0, 0]), epos=np.array([.1, 0, 0]), thickness=.005, lsolid=None, lspace=None,
                 rgba=np.array([1, 0, 0, 1]), type="rect"):
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
    dasharrow_trm = trihelper.gendasharrow(spos=spos, epos=epos, lsolid=lsolid, lspace=lspace, thickness=thickness,
                                           sticktype=type)
    dasharrow_nodepath = dh.trimesh_to_nodepath(dasharrow_trm)
    dasharrow_nodepath.setTransparency(TransparencyAttrib.MDual)
    dasharrow_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    dasharrow_sgm = StaticGeometricModel(dasharrow_nodepath)
    return dasharrow_sgm


def genframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, rgbmatrix=None, alpha=None,
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
    frame_nodepath = NodePath(plotname)
    arrowx_trm = trihelper.genarrow(spos=pos, epos=endx, thickness=thickness)
    arrowx_nodepath = dh.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trihelper.genarrow(spos=pos, epos=endy, thickness=thickness)
    arrowy_nodepath = dh.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = trihelper.genarrow(spos=pos, epos=endz, thickness=thickness)
    arrowz_nodepath = dh.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def genmycframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, alpha=None, plotname="frame"):
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
    return genframe(pos=pos, rotmat=rotmat, length=length, thickness=thickness, rgbmatrix=rgbmatrix, alpha=alpha,
                    plotname=plotname)


def gendashframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3), length=.1, thickness=.005, lsolid=None, lspace=None,
                 rgbmatrix=None, alpha=None, plotname="frame"):
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
    frame_nodepath = NodePath(plotname)
    arrowx_trm = trihelper.gendasharrow(spos=pos, epos=endx, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowx_nodepath = dh.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trihelper.gendasharrow(spos=pos, epos=endy, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowy_nodepath = dh.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = trihelper.gendasharrow(spos=pos, epos=endz, thickness=thickness, lsolid=lsolid, lspace=lspace)
    arrowz_nodepath = dh.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gentorus(axis=np.array([1, 0, 0]), portion=.5, center=np.array([0, 0, 0]), radius=.005, thickness=.0015,
             rgba=np.array([1, 0, 0, 1])):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    torus_trm = trihelper.gentorus(axis=axis, portion=portion, center=center, radius=radius, thickness=thickness)
    torus_nodepath = dh.trimesh_to_nodepath(torus_trm)
    torus_nodepath.setTransparency(TransparencyAttrib.MDual)
    torus_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    torus_sgm = StaticGeometricModel(torus_nodepath)
    return torus_sgm


def gencircarrow(axis=np.array([1, 0, 0]), portion=.5, center=np.array([0, 0, 0]), radius=.05, thickness=.005,
                 rgba=np.array([1, 0, 0, 1])):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    circarrow_trm = trihelper.gencircarrow(axis=axis, portion=portion, center=center, radius=radius,
                                           thickness=thickness)
    circarrow_nodepath = dh.trimesh_to_nodepath(circarrow_trm)
    circarrow_nodepath.setTransparency(TransparencyAttrib.MDual)
    circarrow_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    circarrow_sgm = StaticGeometricModel(circarrow_nodepath)
    return circarrow_sgm


def genpointcloud(verts, colors=np.array([0, 0, 0, .7]), pntsize=5):
    """
    do not use this raw function directly
    use environment.collisionmodel to call it
    gen objmnp
    :param verts:
    :param colors: could be none; specify each point; specify a unified color
    :return:
    """
    colors = np.array(colors)
    pointcloud_nodepath = dh.nodepath_from_points(verts, colors)
    pointcloud_nodepath.setRenderMode(RenderModeAttrib.MPoint, pntsize)
    pointcloud_sgm = StaticGeometricModel(pointcloud_nodepath)
    return pointcloud_sgm


def gensurface(verts, faces, rgba=np.array([1, 0, 0, 1])):
    """
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
    geom = dh.pandageom_from_vvnf(verts, vertnormals, faces)
    node = GeomNode('surface')
    node.addGeom(geom)
    surface_nodepath = NodePath('surface')
    surface_nodepath.attachNewNode(node)
    surface_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    surface_nodepath.setTransparency(TransparencyAttrib.MDual)
    surface_nodepath.setTwoSided(True)
    surface_sgm = StaticGeometricModel(surface_nodepath)
    return surface_sgm


def genpolygon(verts, thickness=0.002, rgba=np.array([0, 0, 0, .7])):
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
    import modeling._pcdhelper as pcdh
    import os
    import math
    import numpy as np
    import basics.robotmath as rm
    import visualization.panda.world as wd

    base = wd.World(camp=[.1, .1, .1], lookatpos=[0, 0, 0])
    this_dir, this_filename = os.path.split(__file__)
    objpath = os.path.join(this_dir, "objects", "bunnysim.stl")
    bunnygm = GeometricModel(objpath)
    bunnygm.setcolor([0.7, 0.7, 0.0, 1.0])
    bunnygm.reparent_to(base)
    bunnygm.showlocalframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnygm.setrotmat(rotmat)

    bunnygm1 = GeometricModel(objpath)
    bunnygm1.setcolor([0.7, 0, 0.7, 1.0])
    bunnygm1.reparent_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnygm1.setpos(np.array([0, .01, 0]))
    bunnygm1.setrotmat(rotmat)

    bunnygm2 = GeometricModel(objpath)
    bunnygm2.setcolor([0, 0.7, 0.7, 1.0])
    bunnygm2.reparent_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnygm1.setpos(np.array([0, .2, 0]))
    bunnygm1.setrotmat(rotmat)

    bunnygmpoints, _ = bunnygm.sample_surface()
    bunnygm1points, _ = bunnygm1.sample_surface()
    bunnygm2points, _ = bunnygm2.sample_surface()
    bpgm = GeometricModel(bunnygmpoints)
    bpgm1 = GeometricModel(bunnygm1points)
    bpgm2 = GeometricModel(bunnygm2points)
    bpgm.reparent_to(base)
    bpgm1.reparent_to(base)
    bpgm2.reparent_to(base)

    lsgm = genlinesegs([np.array([.1,0,.01]), np.array([.01,0,.01]), np.array([.1,0,.1]), np.array([.1,0,.01])])
    lsgm.reparent_to(base)

    gencircarrow(radius=.1, portion=.8).reparent_to(base)
    gendasharrow(spos=np.array([0, 0, 0]), epos=np.array([0, 0, 2])).reparent_to(base)
    gendashframe(pos=np.array([0, 0, 0]), rotmat=np.eye(3)).reparent_to(base)
    axmat = rm.rotmat_from_axangle([1, 1, 1], math.pi/4)
    genframe(rotmat=axmat).reparent_to(base)
    axmat[:, 0] = .1 * axmat[:, 0]
    axmat[:, 1] = .07 * axmat[:, 1]
    axmat[:, 2] = .3 * axmat[:, 2]
    genellipsoid(pos=np.array([0, 0, 0]), axmat=axmat).reparent_to(base)
    print(rm.unit_vector(np.array([0, 0, 0])))

    base.run()
