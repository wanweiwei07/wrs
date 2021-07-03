import os, copy
import basis.data_adapter as da
import basis.trimesh_generator as trihelper
import basis.robot_math as rm
import modeling.model_collection as mc
import numpy as np
import open3d as o3d
from panda3d.core import NodePath, LineSegs, GeomNode, TransparencyAttrib, RenderModeAttrib
from visualization.panda.world import ShowBase
import warnings as wrn


class StaticGeometricModel(object):
    """
    load an object as a static geometric model -> changing pos, rot, color, etc. are not allowed
    there is no extra elements for this model, thus is much faster
    author: weiwei
    date: 20190312
    """

    def __init__(self, initor=None, name="defaultname", btransparency=True, btwosided=False):
        """
        :param initor: path type defined by os.path or trimesh or nodepath
        :param btransparency
        :param name
        """
        if isinstance(initor, StaticGeometricModel):
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._name = copy.deepcopy(initor.name)
            self._localframe = copy.deepcopy(initor.localframe)
        else:
            # make a grandma nodepath to separate decorations (-autoshader) and raw nodepath (+autoshader)
            self._name = name
            self._objpdnp = NodePath(name)
            if isinstance(initor, str):
                self._objpath = initor
                self._objtrm = da.trm.load(self._objpath)
                objpdnp_raw = da.trimesh_to_nodepath(self._objtrm, name='pdnp_raw')
                objpdnp_raw.reparentTo(self._objpdnp)
            elif isinstance(initor, da.trm.Trimesh):
                self._objpath = None
                self._objtrm = initor
                objpdnp_raw = da.trimesh_to_nodepath(self._objtrm)
                objpdnp_raw.reparentTo(self._objpdnp)
            elif isinstance(initor, o3d.geometry.PointCloud):  # TODO should pointcloud be pdnp or pdnp_raw
                self._objpath = None
                self._objtrm = da.trm.Trimesh(np.asarray(initor.points))
                objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices, name='pdnp_raw')
                objpdnp_raw.reparentTo(self._objpdnp)
            elif isinstance(initor, np.ndarray):  # TODO should pointcloud be pdnp or pdnp_raw
                self._objpath = None
                if initor.shape[1] == 3:
                    self._objtrm = da.trm.Trimesh(initor)
                    objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices)
                elif initor.shape[1] == 7:
                    self._objtrm = da.trm.Trimesh(initor[:, :3])
                    objpdnp_raw = da.nodepath_from_points(self._objtrm.vertices, initor[:, 3:].tolist())
                    objpdnp_raw.setRenderMode(RenderModeAttrib.MPoint, 3)
                else:
                    # TODO depth UV?
                    raise NotImplementedError
                objpdnp_raw.reparentTo(self._objpdnp)
            elif isinstance(initor, o3d.geometry.TriangleMesh):
                self._objpath = None
                self._objtrm = da.trm.Trimesh(vertices=initor.vertices, faces=initor.triangles,
                                              face_normals=initor.triangle_normals)
                objpdnp_raw = da.trimesh_to_nodepath(self._objtrm, name='pdnp_raw')
                objpdnp_raw.reparentTo(self._objpdnp)
            elif isinstance(initor, NodePath):
                self._objpath = None
                self._objtrm = None  # TODO nodepath to trimesh?
                objpdnp_raw = initor
                objpdnp_raw.reparentTo(self._objpdnp)
            else:
                self._objpath = None
                self._objtrm = None
                objpdnp_raw = NodePath("pdnp_raw")
                objpdnp_raw.reparentTo(self._objpdnp)
            if btransparency:
                self._objpdnp.setTransparency(TransparencyAttrib.MDual)
            if btwosided:
                self._objpdnp.getChild(0).setTwoSided(True)
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
    def objpdnp(self):
        # read-only property
        return self._objpdnp

    @property
    def objpdnp_raw(self):
        # read-only property
        return self._objpdnp.getChild(0)

    @property
    def objtrm(self):
        # read-only property
        # 20210328 comment out, allow None
        # if self._objtrm is None:
        #     raise ValueError("Only applicable to models with a trimesh!")
        return self._objtrm

    @property
    def localframe(self):
        # read-only property
        return self._localframe

    @property
    def volume(self):
        # read-only property
        if self._objtrm is None:
            raise ValueError("Only applicable to models with a trimesh!")
        return self._objtrm.volume

    def set_rgba(self, rgba):
        self._objpdnp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def get_rgba(self):
        return da.pdv4_to_npv4(self._objpdnp.getColor())  # panda3d.core.LColor -> LBase4F

    def clear_rgba(self):
        self._objpdnp.clearColor()

    def set_scale(self, scale=[1, 1, 1]):
        self._objpdnp.setScale(scale[0], scale[1], scale[2])
        self._objtrm.apply_scale(scale)

    def get_scale(self):
        return da.pdv3_to_npv3(self._objpdnp.getScale())

    def set_vert_size(self, size=.005):
        self.objpdnp_raw.setRenderModeThickness(size * 1000)

    def attach_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, StaticGeometricModel):  # prepared for decorations like local frames
            self._objpdnp.reparentTo(obj.objpdnp)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_gm(self)
        else:
            print(
                "Must be ShowBase, modeling.StaticGeometricModel, GeometricModel, CollisionModel, or CollisionModelCollection!")

    def detach(self):
        self._objpdnp.detachNode()

    def remove(self):
        self._objpdnp.removeNode()

    def show_localframe(self):
        self._localframe = gen_frame()
        self._localframe.attach_to(self)

    def unshow_localframe(self):
        if self._localframe is not None:
            self._localframe.remove()
            self._localframe = None

    def copy(self):
        return copy.deepcopy(self)


class WireFrameModel(StaticGeometricModel):

    def __init__(self, initor=None, name="auto"):
        """
        :param initor: path type defined by os.path or trimesh or nodepath
        """
        super().__init__(initor=initor, btransparency=False, name=name)
        self.objpdnp_raw.setRenderModeWireframe()
        self.objpdnp_raw.setLightOff()
        # self.set_rgba(rgba=[0,0,0,1])

    # suppress functions
    def __getattr__(self, attr_name):
        if attr_name == 'sample_surface':
            raise AttributeError("Wireframe Model does not support sampling surface!")
        return getattr(self._wrapped, attr_name)

    @property
    def name(self):
        # read-only property
        return self._name

    @property
    def objpath(self):
        # read-only property
        return self._objpath

    @property
    def objpdnp(self):
        # read-only property
        return self._objpdnp

    @property
    def objpdnp_raw(self):
        # read-only property
        return self._objpdnp.getChild(0)

    @property
    def objtrm(self):
        # read-only property
        if self._objtrm is None:
            raise ValueError("Only applicable to models with a trimesh!")
        return self._objtrm

    @property
    def localframe(self):
        # read-only property
        return self._localframe

    @property
    def volume(self):
        # read-only property
        if self._objtrm is None:
            raise ValueError("Only applicable to models with a trimesh!")
        return self._objtrm.volume

    def set_rgba(self, rgba):
        wrn.warn("Right not the set_rgba fn for a WireFrame instance is not implemented!")
        # self._objpdnp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def get_rgba(self):
        return da.pdv4_to_npv4(self._objpdnp.getColor())  # panda3d.core.LColor -> LBase4F

    def clear_rgba(self):
        self._objpdnp.clearColor()

    def set_scale(self, scale=[1, 1, 1]):
        self._objpdnp.setScale(scale[0], scale[1], scale[2])

    def get_scale(self):
        return da.pdv3_to_npv3(self._objpdnp.getScale())

    def attach_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, StaticGeometricModel):  # prepared for decorations like local frames
            self._objpdnp.reparentTo(obj.objpdnp)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_gm(self)
        else:
            print("Must be ShowBase, modeling.StaticGeometricModel, GeometricModel, "
                  "CollisionModel, or CollisionModelCollection!")

    def detach(self):
        self._objpdnp.detachNode()

    def remove(self):
        self._objpdnp.removeNode()

    def show_localframe(self):
        self._localframe = gen_frame()
        self._localframe.attach_to(self)

    def unshow_localframe(self):
        if self._localframe is not None:
            self._localframe.removeNode()
            self._localframe = None


class GeometricModel(StaticGeometricModel):
    """
    load an object as a geometric model
    there is no extra elements for this model, thus is much faster
    author: weiwei
    date: 20190312
    """

    def __init__(self, initor=None, name="defaultname", btransparency=True, btwosided=False):
        """
        :param initor: path type defined by os.path or trimesh or nodepath
        """
        if isinstance(initor, GeometricModel):
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._name = copy.deepcopy(initor.name)
            self._localframe = copy.deepcopy(initor.localframe)
        else:
            super().__init__(initor=initor, name=name, btransparency=btransparency, btwosided=btwosided)
        self.objpdnp_raw.setShaderAuto()

    def set_pos(self, npvec3):
        self._objpdnp.setPos(npvec3[0], npvec3[1], npvec3[2])

    def get_pos(self):
        return da.pdv3_to_npv3(self._objpdnp.getPos())

    def set_rotmat(self, npmat3):
        self._objpdnp.setQuat(da.npmat3_to_pdquat(npmat3))

    def get_rotmat(self):
        return da.pdquat_to_npmat3(self._objpdnp.getQuat())

    def set_homomat(self, npmat4):
        self._objpdnp.setPosQuat(da.npv3_to_pdv3(npmat4[:3, 3]), da.npmat3_to_pdquat(npmat4[:3, :3]))

    def get_homomat(self):
        npv3 = da.pdv3_to_npv3(self._objpdnp.getPos())
        npmat3 = da.pdquat_to_npmat3(self._objpdnp.getQuat())
        return rm.homomat_from_posrot(npv3, npmat3)

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
        npmat3 = rm.rotmat_from_euler(roll, pitch, yaw, axes="sxyz")
        self.set_rotmat(npmat3)

    def get_rpy(self):
        """
        get the pose of the object using rpy
        :return: [r, p, y] in radian
        author: weiwei
        date: 20190513
        """
        npmat3 = self.get_rotmat()
        rpy = rm.rotmat_to_euler(npmat3, axes="sxyz")
        return np.array([rpy[0], rpy[1], rpy[2]])

    def set_transparency(self, attribute):
        return self._objpdnp.setTransparency(attribute)

    def sample_surface(self, radius=0.005, nsample=None, toggle_option='face_ids'):
        """
        :param raidus:
        :param toggle_option; 'face_ids', 'normals', None
        :return:
        author: weiwei
        date: 20191228
        """
        if self._objtrm is None:
            raise ValueError("Only applicable to models with a trimesh!")
        if nsample is None:
            nsample = int(round(self.objtrm.area / ((radius * 0.3) ** 2)))
        points, face_ids = self.objtrm.sample_surface(nsample, radius=radius, toggle_faceid=True)
        # transform
        points = rm.homomat_transform_points(self.get_homomat(), points)
        if toggle_option is None:
            return np.array(points)
        elif toggle_option == 'face_ids':
            return np.array(points), np.array(face_ids)
        elif toggle_option == 'normals':
            return np.array(points), rm.homomat_transform_points(self.get_homomat(), self.objtrm.face_normals[face_ids])
        else:
            print("toggle_option must be None, point_face_ids, or point_nromals")

    def copy(self):
        return copy.deepcopy(self)


## primitives are stationarygeometric model, once defined, they cannot be changed
# TODO: further decouple from Panda trimesh->staticgeometricmodel
def gen_linesegs(linesegs, thickness=0.001, rgba=[0, 0, 0, 1]):
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
    M_TO_PIXEL = 3779.53
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * M_TO_PIXEL)
    ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    for p0p1tuple in linesegs:
        ls.moveTo(p0p1tuple[0][0], p0p1tuple[0][1], p0p1tuple[0][2])
        ls.drawTo(p0p1tuple[1][0], p0p1tuple[1][1], p0p1tuple[1][2])
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


# def gen_linesegs(verts, thickness=0.005, rgba=[0,0,0,1]):
#     """
#     gen continuous linsegs
#     :param verts: nx3 list, each nearby pair will be used to draw one segment, defined in a local 0 frame
#     :param rgba:
#     :param thickness:
#     :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
#     :return: a geomtric model
#     author: weiwei
#     date: 20161216
#     """
#     segs = LineSegs()
#     segs.setThickness(thickness * 1000.0)
#     segs.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
#     for i in range(len(verts) - 1):
#         tmpstartvert = verts[i]
#         tmpendvert = verts[i + 1]
#         segs.moveTo(tmpstartvert[0], tmpstartvert[1], tmpstartvert[2])
#         segs.drawTo(tmpendvert[0], tmpendvert[1], tmpendvert[2])
#     lsnp = NodePath('linesegs')
#     lsnp.attachNewNode(segs.create())
#     lsnp.setTransparency(TransparencyAttrib.MDual)
#     ls_sgm = StaticGeometricModel(lsnp)
#     return ls_sgm


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=[1, 0, 0, 1], subdivisions=2):
    """
    :param pos:
    :param radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_trm = trihelper.gen_sphere(pos, radius, subdivisions)
    sphere_sgm = StaticGeometricModel(sphere_trm)
    sphere_sgm.set_rgba(rgba)
    return sphere_sgm


def gen_ellipsoid(pos=np.array([0, 0, 0]),
                  axmat=np.eye(3),
                  rgba=[1, 1, 0, .3]):
    """
    :param pos:
    :param axmat: 3x3 mat, each column is an axis of the ellipse
    :param rgba:
    :return:
    author: weiwei
    date: 20200701osaka
    """
    ellipsoid_trm = trihelper.gen_ellipsoid(pos=pos, axmat=axmat)
    ellipsoid_sgm = StaticGeometricModel(ellipsoid_trm)
    ellipsoid_sgm.set_rgba(rgba)
    return ellipsoid_sgm


def gen_stick(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              thickness=.005, type="rect",
              rgba=[1, 0, 0, 1], sections=8):
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
    stick_trm = trihelper.gen_stick(spos=spos, epos=epos, thickness=thickness, type=type, sections=sections)
    stick_sgm = StaticGeometricModel(stick_trm)
    stick_sgm.set_rgba(rgba)
    return stick_sgm


def gen_dashstick(spos=np.array([0, 0, 0]),
                  epos=np.array([.1, 0, 0]),
                  thickness=.005,
                  lsolid=None,
                  lspace=None,
                  rgba=[1, 0, 0, 1],
                  type="rect"):
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
    dashstick_trm = trihelper.gen_dashstick(spos=spos,
                                            epos=epos,
                                            lsolid=lsolid,
                                            lspace=lspace,
                                            thickness=thickness,
                                            sticktype=type)
    dashstick_sgm = StaticGeometricModel(dashstick_trm)
    dashstick_sgm.set_rgba(rgba=rgba)
    return dashstick_sgm


def gen_box(extent=np.array([1, 1, 1]),
            homomat=np.eye(4),
            rgba=[1, 0, 0, 1]):
    """
    :param extent:
    :param homomat:
    :return:
    author: weiwei
    date: 20191229osaka
    """
    box_trm = trihelper.gen_box(extent=extent, homomat=homomat)
    box_sgm = StaticGeometricModel(box_trm)
    box_sgm.set_rgba(rgba=rgba)
    return box_sgm


def gen_dumbbell(spos=np.array([0, 0, 0]),
                 epos=np.array([.1, 0, 0]),
                 thickness=.005,
                 rgba=[1, 0, 0, 1]):
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
    dumbbell_sgm.set_rgba(rgba=rgba)
    return dumbbell_sgm


def gen_cone(spos=np.array([0, 0, 0]),
             epos=np.array([0.1, 0, 0]),
             rgba=np.array([.7, .7, .7, .3]),
             radius=0.005,
             sections=8):
    """
    :param spos:
    :param epos:
    :param radius:
    :param sections:
    :return:
    author: weiwei
    date: 20210625
    """
    cone_trm = trihelper.gen_cone(spos=spos, epos=epos, radius=radius, sections=sections)
    cone_sgm = GeometricModel(cone_trm)
    cone_sgm.set_rgba(rgba=rgba)
    return cone_sgm

def gen_arrow(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              thickness=.005, rgba=[1, 0, 0, 1],
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
    arrow_sgm.set_rgba(rgba=rgba)
    return arrow_sgm


def gen_dasharrow(spos=np.array([0, 0, 0]),
                  epos=np.array([.1, 0, 0]),
                  thickness=.005, lsolid=None,
                  lspace=None,
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
    dasharrow_trm = trihelper.gen_dasharrow(spos=spos,
                                            epos=epos,
                                            lsolid=lsolid,
                                            lspace=lspace,
                                            thickness=thickness,
                                            sticktype=type)
    dasharrow_sgm = StaticGeometricModel(dasharrow_trm)
    dasharrow_sgm.set_rgba(rgba=rgba)
    return dasharrow_sgm


def gen_frame(pos=np.array([0, 0, 0]),
              rotmat=np.eye(3),
              length=.1,
              thickness=.005,
              rgbmatrix=None,
              alpha=None,
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


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=.5,
              center=np.array([0, 0, 0]),
              radius=.005,
              thickness=.0015,
              rgba=[1, 0, 0, 1],
              sections=8,
              discretization=24):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    torus_trm = trihelper.gen_torus(axis=axis,
                                    starting_vector=starting_vector,
                                    portion=portion,
                                    center=center,
                                    radius=radius,
                                    thickness=thickness,
                                    sections=sections,
                                    discretization=discretization)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_rgba(rgba=rgba)
    return torus_sgm


def gen_dashtorus(axis=np.array([1, 0, 0]),
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  radius=0.1,
                  thickness=0.005,
                  rgba=[1,0,0,1],
                  lsolid=None,
                  lspace=None,
                  sections=8,
                  discretization=24):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    torus_trm = trihelper.gen_dashtorus(axis=axis,
                                        portion=portion,
                                        center=center,
                                        radius=radius,
                                        thickness=thickness,
                                        lsolid=lsolid,
                                        lspace=lspace,
                                        sections=sections,
                                        discretization=discretization)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_rgba(rgba=rgba)
    return torus_sgm


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  radius=.05,
                  thickness=.005,
                  rgba=[1, 0, 0, 1],
                  sections=8,
                  discretization=24):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    circarrow_trm = trihelper.gen_circarrow(axis=axis,
                                            starting_vector=starting_vector,
                                            portion=portion,
                                            center=center,
                                            radius=radius,
                                            thickness=thickness,
                                            sections=sections,
                                            discretization=discretization)
    circarrow_sgm = StaticGeometricModel(circarrow_trm)
    circarrow_sgm.set_rgba(rgba=rgba)
    return circarrow_sgm


def gen_pointcloud(points, rgbas=[[0, 0, 0, .7]], pntsize=3):
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


def gen_submesh(verts, faces, rgba=[1, 0, 0, 1]):
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

def gen_frame_box(extent=[.02, .02, .02], homomat=np.eye(4), rgba=[0, 0, 0, 1], thickness=.001):
    """
    draw a 3D box, only show edges
    :param extent:
    :param homomat:
    :return:
    """
    M_TO_PIXEL = 3779.53
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * M_TO_PIXEL)
    ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    center_pos = homomat[:3,3]
    x_axis = homomat[:3,0]
    y_axis = homomat[:3,1]
    z_axis = homomat[:3,2]
    x_min, x_max = -x_axis*extent[0]/2, x_axis*extent[0]/2
    y_min, y_max = -y_axis*extent[1]/2, y_axis*extent[1]/2
    z_min, z_max = -z_axis*extent[2]/2, z_axis*extent[2]/2
    # max, max, max
    print(center_pos+np.array([x_max, y_max, z_max]))
    ls.moveTo(da.npv3_to_pdv3(center_pos+x_max+y_max+z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_max+y_max+z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_max+y_min+z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_max+y_min+z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_max+y_max+z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_max+z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_min+z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_min+z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_max+z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_max+z_max))
    ls.moveTo(da.npv3_to_pdv3(center_pos+x_max+y_max+z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_max+z_min))
    ls.moveTo(da.npv3_to_pdv3(center_pos+x_max+y_min+z_min))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_min+z_min))
    ls.moveTo(da.npv3_to_pdv3(center_pos+x_max+y_min+z_max))
    ls.drawTo(da.npv3_to_pdv3(center_pos+x_min+y_min+z_max))
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm

def gen_surface(surface_callback, rng, granularity=.01):
    surface_trm = trihelper.gen_surface(surface_callback, rng, granularity)
    surface_gm = GeometricModel(surface_trm, btwosided=True)
    return surface_gm


if __name__ == "__main__":
    import os
    import math
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnygm = GeometricModel(objpath)
    bunnygm.set_rgba([0.7, 0.7, 0.0, 1.0])
    bunnygm.attach_to(base)
    bunnygm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnygm.set_rotmat(rotmat)

    bunnygm1 = bunnygm.copy()
    bunnygm1.set_rgba([0.7, 0, 0.7, 1.0])
    bunnygm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnygm1.set_pos(np.array([0, .01, 0]))
    bunnygm1.set_rotmat(rotmat)

    bunnygm2 = bunnygm1.copy()
    bunnygm2.set_rgba([0, 0.7, 0.7, 1.0])
    bunnygm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnygm2.set_pos(np.array([0, .2, 0]))
    bunnygm2.set_rotmat(rotmat)
    bunnygm2.set_scale([2, 1, 3])

    bunnygmpoints, _ = bunnygm.sample_surface()
    bunnygm1points, _ = bunnygm1.sample_surface()
    bunnygm2points, _ = bunnygm2.sample_surface()
    bpgm = GeometricModel(bunnygmpoints)
    bpgm1 = GeometricModel(bunnygm1points)
    bpgm2 = GeometricModel(bunnygm2points)
    bpgm.attach_to(base)
    bpgm.set_scale([2, 1, 3])
    bpgm.set_vert_size(.01)
    bpgm1.attach_to(base)
    bpgm2.attach_to(base)

    lsgm = gen_linesegs([[np.array([.1, 0, .01]), np.array([.01, 0, .01])],
                         [np.array([.01, 0, .01]), np.array([.1, 0, .1])],
                         [np.array([.1, 0, .1]), np.array([.1, 0, .01])]])
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

    pos= np.array([.3,0,0])
    rotmat = rm.rotmat_from_euler(math.pi/6,0,0)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    gen_frame_box([.1, .2, .3], homomat).attach_to(base)

    base.run()
