import os, copy
import basis.data_adapter as da
import basis.trimesh_factory as trm_factory
import basis.robot_math as rm
import basis.constant as cst
import modeling.model_collection as mc
import numpy as np
import open3d as o3d
from panda3d.core import NodePath, LineSegs, GeomNode, TransparencyAttrib, RenderModeAttrib
from visualization.panda.world import ShowBase
import warnings as wrn


class StaticGeometricModel(object):
    """
    load an object as a static geometric model -> changing pos, rotmat, color, etc. are not allowed
    there is no extra elements for this model, thus is much faster
    author: weiwei
    date: 20190312, 20230812
    """

    def __init__(self,
                 initor=None,
                 name="sgm",
                 toggle_transparency=True,
                 toggle_twosided=False):
        """
        :param initor: path end_type defined by os.path or trimesh or pdndp
        :param toggle_transparency
        :param name
        """
        if isinstance(initor, StaticGeometricModel):
            self._file_path = copy.deepcopy(initor.file_path)
            self._trm_mesh = copy.deepcopy(initor.trm_mesh)
            self._pdndp = copy.deepcopy(initor.pdndp)
            self._name = copy.deepcopy(initor.name)
            self._local_frame = copy.deepcopy(initor.local_frame)
        else:
            # make a grandma pdndp to separate decorations (-autoshader) and raw pdndp (+autoshader)
            self._name = name
            self._pdndp = NodePath(name)
            if isinstance(initor, str):
                self._file_path = initor
                self._trm_mesh = da.trm.load(self._file_path)
                pdndp_core = da.trimesh_to_nodepath(self._trm_mesh, name='pdndp_core')
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, da.trm.Trimesh):
                self._file_path = None
                self._trm_mesh = initor
                pdndp_core = da.trimesh_to_nodepath(self._trm_mesh)
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, o3d.geometry.PointCloud):  # TODO should pointcloud be pdndp or pdnp_raw
                self._file_path = None
                self._trm_mesh = da.trm.Trimesh(np.asarray(initor.points))
                pdndp_core = da.pdgeomndp_from_v(self._trm_mesh.vertices, name='pdndp_core')
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, np.ndarray):  # TODO should pointcloud be pdndp or pdnp_raw
                self._file_path = None
                if initor.ndim == 2:
                    if initor.shape[1] == 3:
                        self._trm_mesh = da.trm.Trimesh(initor)
                        pdndp_core = da.pdgeomndp_from_v(self._trm_mesh.vertices)
                        pdndp_core.setRenderModeThickness(.001 * da.M_TO_PIXEL)
                    elif initor.shape[1] == 7:
                        self._trm_mesh = da.trm.Trimesh(initor[:, :3])
                        pdndp_core = da.pdgeomndp_from_v(self._trm_mesh.vertices, initor[:, 3:])
                        pdndp_core.setRenderModeThickness(.001 * da.M_TO_PIXEL)
                    else:
                        # TODO depth UV?
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, o3d.geometry.TriangleMesh):
                self._file_path = None
                self._trm_mesh = da.trm.Trimesh(vertices=initor.vertices, faces=initor.triangles,
                                                face_normals=initor.triangle_normals)
                pdndp_core = da.trimesh_to_nodepath(self._trm_mesh, name='pdndp_core')
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, NodePath):  # TODO: deprecate 20230815
                self._file_path = None
                self._trm_mesh = None
                pdndp_core = initor
                pdndp_core.reparentTo(self._pdndp)
            else:  # empty model
                self._file_path = None
                self._trm_mesh = None
                pdndp_core = NodePath("pdndp_core")
                pdndp_core.reparentTo(self._pdndp)
            if toggle_transparency:
                self._pdndp.setTransparency(TransparencyAttrib.MDual)
            if toggle_twosided:
                self._pdndp.getChild(0).setTwoSided(True)
            self._local_frame = None

    @property
    def name(self):
        # read-only property
        return self._name

    @property
    def file_path(self):
        # read-only property
        return self._file_path

    @property
    def pdndp(self):
        # read-only property
        return self._pdndp

    @property
    def pdndp_core(self):
        """
        read-only property
        for rendering purpose
        i.e. frames will be attached to pdndp and will not be influenced by rendering changes made to pdndp_core
        :return:
        """
        return self._pdndp.getChild(0)

    @property
    def trm_mesh(self):
        # read-only property
        # 20210328 comment out, allow None
        # if self._trm_mesh is None:
        #     raise ValueError("Only applicable to models with a trimesh!")
        return self._trm_mesh

    @property
    def local_frame(self):
        # read-only property
        return self._local_frame

    @property
    def volume(self):
        # read-only property
        if self._trm_mesh is None:
            raise ValueError("Only applicable to models with a trimesh!")
        return self._trm_mesh.volume

    def set_rgba(self, rgba):
        self._pdndp.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

    def set_alpha(self, alpha):
        rgba = self._pdndp.getColor()
        self._pdndp.setColor(rgba[0], rgba[1], rgba[2], alpha)

    def set_scale(self, scale=np.array([1, 1, 1])):
        """
        :param scale: 1x3 nparray, each element denotes the scale in x, y, and z dimensions
        :return:
        """
        self._pdndp.setScale(*scale)
        self._trm_mesh.apply_scale(scale)

    def set_point_size(self, size=.001):
        # only applicable to point clouds
        self.pdndp_core.setRenderModeThickness(size * da.M_TO_PIXEL)

    def get_rgba(self):
        return da.pdvec4_to_npvec4(self._pdndp.getColor())  # panda3d.core.LColor -> LBase4F

    def clear_rgba(self):
        self._pdndp.clearColor()

    def attach_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._pdndp.reparentTo(target.render)
        elif isinstance(target, StaticGeometricModel):  # prepared for decorations like local frames
            self._pdndp.reparentTo(target.pdndp)
        elif isinstance(target, mc.ModelCollection):
            target.add_gm(self)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, ModelCollection!")

    def detach(self):
        self._pdndp.detachNode()

    def remove(self):
        self._pdndp.removeNode()

    def show_local_frame(self):
        self._local_frame = gen_frame()
        self._local_frame.attach_to(self)

    def unshow_local_frame(self):
        if self._local_frame is not None:
            self._local_frame.remove()
            self._local_frame = None

    def copy(self):
        return copy.deepcopy(self)


class WireFrameModel(StaticGeometricModel):

    def __init__(self,
                 initor=None,
                 name="wsgm"):
        """
        :param initor: path end_type defined by os.path or trimesh or pdndp
        """
        super().__init__(initor=initor, toggle_transparency=False, name=name)
        # apply rendering effects to pdndp_core
        # frames will be attached to pdndp and will not be influenced by changes made to pdndp_core
        self.pdndp_core.setRenderModeWireframe()
        self.pdndp_core.setLightOff()
        # self.set_rgba(rgba=[0,0,0,1])

    def set_rgba(self, rgba):
        wrn.warn("Right not the set_rgba fn for a WireFrame instance is not implemented!")


def update_delay(argument):
    """
    :param argument: "geometry", "cdprimitive", or "cdmesh"
    :return:
    author: weiwei
    date: 20231122
    """
    def update_delay_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self.is_delayed[argument]:
                self._pdndp.setPosQuat(da.npvec3_to_pdvec3(self.pos), da.npmat3_to_pdquat(self.rotmat))
            return method(self, *args, **kwargs)
        return wrapper


class GeometricModel(StaticGeometricModel):
    """
    load an object as a geometric model
    there is no extra elements for this model, thus is much faster
    author: weiwei
    date: 20190312
    """

    def __init__(self,
                 initor=None,
                 name="mgm",
                 toggle_transparency=True,
                 toggle_twosided=False):
        """
        :param initor: path end_type defined by os.path or trimesh or pdndp
        """
        if isinstance(initor, GeometricModel):
            self._file_path = copy.deepcopy(initor.file_path)
            self._trm_mesh = copy.deepcopy(initor.trm_mesh)
            self._pdndp = copy.deepcopy(initor.pdndp)
            self._name = copy.deepcopy(initor.name)
            self._local_frame = copy.deepcopy(initor.local_frame)
            self._pos = copy.deepcopy(initor._pos)
            self._rotmat = copy.deepcopy(initor._rotmat)
            self._is_delayed=copy.deepcopy(initor._is_delayed)
        else:
            super().__init__(initor=initor,
                             name=name,
                             toggle_transparency=toggle_transparency,
                             toggle_twosided=toggle_twosided)
            self._pos = np.zeros(3)
            self._rotmat = np.eye(3)
            self._is_delayed = {"geometry": False}
        self.pdndp_core.setShaderAuto()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, val: np.ndarray):
        self._pos = val
        self._is_delayed = True

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    def rotmat(self, val: np.ndarray):
        self._rotmat = val
        self._is_delayed = True

    @property
    def homomat(self):
        homomat = np.eye(4)
        homomat[:3, :3] = self._rotmat
        homomat[:3, 3] = self._pos
        return homomat

    @homomat.setter
    def homomat(self, val: np.ndarray):
        self._pos = val[:3, 3]
        self._rotmat = val[:3, :3]
        self._is_delayed = True

    @property
    def is_delayed(self):
        return self._is_delayed

    # def set_pos(self, npvec3):
    #     self._pdndp.setPos(*npvec3)
    #
    # def set_rotmat(self, npmat3):
    #     self._pdndp.setQuat(da.npmat3_to_pdquat(npmat3))
    #
    # def set_homomat(self, npmat4):
    #     self._pdndp.setPosQuat(da.npvec3_to_pdvec3(npmat4[:3, 3]), da.npmat3_to_pdquat(npmat4[:3, :3]))
    #
    # def get_pos(self):
    #     return da.pdvec3_to_npvec3(self._pdndp.getPos())
    #
    # def get_rotmat(self):
    #     return da.pdquat_to_npmat3(self._pdndp.getQuat())
    #
    # def get_homomat(self):
    #     npv3 = da.pdvec3_to_npvec3(self._pdndp.getPos())
    #     npmat3 = da.pdquat_to_npmat3(self._pdndp.getQuat())
    #     return rm.homomat_from_posrot(npv3, npmat3)

    def set_transparency(self, attribute):
        return self._pdndp.setTransparency(attribute)

    @update_geom_delay_decorator
    def attach_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._pdndp.reparentTo(target.render)
        elif isinstance(target, StaticGeometricModel):  # prepared for decorations like local frames
            self._pdndp.reparentTo(target.pdndp)
        elif isinstance(target, mc.ModelCollection):
            target.add_gm(self)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, ModelCollection!")

    def sample_surface(self, radius=0.005, n_samples=None, toggle_option=None):
        """
        :param raidus:
        :param toggle_option; 'face_ids', 'normals', None
        :return:
        author: weiwei
        date: 20191228
        """
        if self._trm_mesh is None:
            raise ValueError("Only applicable to models with a trimesh!")
        if n_samples is None:
            n_samples = int(round(self.trm_mesh.area / ((radius * 0.3) ** 2)))
        points, face_ids = self.trm_mesh.sample_surface(n_samples, radius=radius, toggle_faceid=True)
        # transform
        points = rm.transform_points_by_homomat(self.homomat, points)
        if toggle_option is None:
            return np.array(points)
        elif toggle_option == 'face_ids':
            return np.array(points), np.array(face_ids)
        elif toggle_option == 'normals':
            return np.array(points), rm.transform_points_by_homomat(self.homomat,
                                                                    self.trm_mesh.face_normals[face_ids])
        else:
            print("The toggle_option parameter must be \"None\", \"point_face_ids\", or \"point_nromals\"!")

    def copy(self):
        return copy.deepcopy(self)


def gen_linesegs(linesegs,
                 thickness=0.001,
                 rgba=np.array([0, 0, 0, 1])):
    """
    gen linsegs -- non-continuous segs are allowed
    :param linesegs: nx2x3 nparray, defined in local 0 frame
    :param rgba:
    :param thickness:
    :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
    :return: a geomtric model
    author: weiwei
    date: 20161216, 20201116, 20230812
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(*rgba)
    for p0_p1_tuple in linesegs:
        ls.moveTo(*p0_p1_tuple[0])
        ls.drawTo(*p0_p1_tuple[1])
    # Create and return a node with the segments
    ls_pdndp = NodePath(ls.create())
    ls_pdndp.setTransparency(TransparencyAttrib.MDual)
    ls_pdndp.setLightOff()
    ls_sgm = StaticGeometricModel(initor=ls_pdndp)
    return ls_sgm


def gen_sphere(pos=np.array([0, 0, 0]),
               radius=0.0015,
               rgba=np.array([1, 0, 0, 1]),
               ico_level=2):
    """
    :param pos:
    :param radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_trm = trm_factory.gen_sphere(pos=pos, radius=radius, ico_level=ico_level)
    sphere_sgm = StaticGeometricModel(initor=sphere_trm)
    sphere_sgm.set_rgba(rgba=rgba)
    return sphere_sgm


def gen_ellipsoid(pos=np.array([0, 0, 0]),
                  axes_mat=np.eye(3),
                  rgba=np.array([1, 1, 0, .3])):
    """
    :param pos:
    :param axes_mat: 3x3 mat, each column is an axis of the ellipse
    :param rgba:
    :return:
    author: weiwei
    date: 20200701osaka
    """
    ellipsoid_trm = trm_factory.gen_ellipsoid(pos=pos, axmat=axes_mat)
    ellipsoid_sgm = StaticGeometricModel(initor=ellipsoid_trm)
    ellipsoid_sgm.set_rgba(rgba=rgba)
    return ellipsoid_sgm


def gen_stick(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              radius=.0025,
              rgba=np.array([1, 0, 0, 1]),
              type="rect",
              n_sec=18):
    """
    :param spos:
    :param epos:
    :param radius:
    :param rgba:
    :param type: rect or round
    :param n_sec:
    :return:
    author: weiwei
    date: 20191229osaka
    """
    stick_trm = trm_factory.gen_stick(spos=spos, epos=epos, radius=radius, type=type, n_sec=n_sec)
    stick_sgm = StaticGeometricModel(initor=stick_trm)
    stick_sgm.set_rgba(rgba=rgba)
    return stick_sgm


def gen_dashed_stick(spos=np.array([0, 0, 0]),
                     epos=np.array([.1, 0, 0]),
                     radius=.0025,
                     rgba=np.array([1, 0, 0, 1]),
                     len_solid=None,
                     len_interval=None,
                     type="rect",
                     n_sec=18):
    """
    :param spos:
    :param epos:
    :param radius:
    :param len_solid: axis_length of the solid section, 1*major_radius by default
    :param len_interval: axis_length of the interval between two solids, 1.5*major_radius by default
    :param rgba:
    :return:
    author: weiwei
    date: 20200625osaka
    """
    dashstick_trm = trm_factory.gen_dashstick(spos=spos,
                                              epos=epos,
                                              len_solid=len_solid,
                                              len_interval=len_interval,
                                              radius=radius,
                                              type=type,
                                              n_sec=n_sec)
    dashstick_sgm = StaticGeometricModel(initor=dashstick_trm)
    dashstick_sgm.set_rgba(rgba=rgba)
    return dashstick_sgm


# def gen_box(xyz_lengths=np.array([1, 1, 1]),
#             pos=np.eye(4),
#             rgba=np.array([1, 0, 0, 1])):
#     """
#     :param xyz_lengths:
#     :param pos:
#     :param rgba:
#     :return:
#     author: weiwei
#     date: 20191229osaka
#     """
#     box_trm = trm_factory.gen_box(xyz_lengths=xyz_lengths, pos=pos)
#     box_sgm = StaticGeometricModel(initializer=box_trm)
#     box_sgm.set_rgba(rgba=rgba)
#     return box_sgm


def gen_box(xyz_lengths=np.array([1, 1, 1]),
            pos=np.zeros(3),
            rotmat=np.eye(3),
            rgba=np.array([1, 0, 0, 1])):
    """
    :param xyz_lengths:
    :param pos:
    :param rotmat:
    :param rgba:
    :return:
    author: weiwei
    date: 20191229osaka, 20230830
    """
    box_trm = trm_factory.gen_box(xyz_lengths=xyz_lengths, pos=pos, rotmat=rotmat)
    box_sgm = StaticGeometricModel(initor=box_trm)
    box_sgm.set_rgba(rgba=rgba)
    return box_sgm


def gen_dumbbell(spos=np.array([0, 0, 0]),
                 epos=np.array([.1, 0, 0]),
                 rgba=np.array([1, 0, 0, 1]),
                 stick_radius=.0025,
                 n_sec=18,
                 sphere_radius=None,
                 sphere_ico_level=2):
    """
    :param sphere_radius:
    :param spos:
    :param epos:
    :param stick_radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    dumbbell_trm = trm_factory.gen_dumbbell(spos=spos,
                                            epos=epos,
                                            stick_radius=stick_radius,
                                            n_sec=n_sec,
                                            sphere_radius=sphere_radius,
                                            sphere_ico_level=sphere_ico_level)
    dumbbell_sgm = StaticGeometricModel(dumbbell_trm)
    dumbbell_sgm.set_rgba(rgba=rgba)
    return dumbbell_sgm


def gen_cone(spos=np.array([0, 0, 0]),
             epos=np.array([0.1, 0, 0]),
             rgba=np.array([.7, .7, .7, .3]),
             bottom_radius=0.005,
             n_sec=8):
    """
    :param spos:
    :param epos:
    :param bottom_radius:
    :param n_sec:
    :return:
    author: weiwei
    date: 20210625
    """
    cone_trm = trm_factory.gen_cone(spos=spos, epos=epos, bottom_radius=bottom_radius, n_sec=n_sec)
    cone_sgm = GeometricModel(cone_trm)
    cone_sgm.set_rgba(rgba=rgba)
    return cone_sgm


def gen_arrow(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              rgba=np.array([1, 0, 0, 1]),
              stick_radius=.0025,
              stick_type="rect"):
    """
    :param spos:
    :param epos:
    :param stick_radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20200115osaka
    """
    arrow_trm = trm_factory.gen_arrow(spos=spos, epos=epos, stick_radius=stick_radius, stick_type=stick_type)
    arrow_sgm = StaticGeometricModel(arrow_trm)
    arrow_sgm.set_rgba(rgba=rgba)
    return arrow_sgm


def gen_dashed_arrow(spos=np.array([0, 0, 0]),
                     epos=np.array([.1, 0, 0]),
                     rgba=np.array([1, 0, 0, 1]),
                     stick_radius=.0025,
                     len_solid=None,
                     len_interval=None,
                     type="rect"):
    """
    :param spos:
    :param epos:
    :param stick_radius:
    :param len_solid: axis_length of the solid section, 1*major_radius by default
    :param len_interval: axis_length of the empty section, 1.5*major_radius by default
    :param rgba:
    :return:
    author: weiwei
    date: 20200625osaka
    """
    dasharrow_trm = trm_factory.gen_dasharrow(spos=spos,
                                              epos=epos,
                                              len_solid=len_solid,
                                              len_interval=len_interval,
                                              stick_radius=stick_radius,
                                              stick_type=type)
    dasharrow_sgm = StaticGeometricModel(dasharrow_trm)
    dasharrow_sgm.set_rgba(rgba=rgba)
    return dasharrow_sgm


def gen_frame(pos=np.array([0, 0, 0]),
              rotmat=np.eye(3),
              axis_length=.1,
              axis_radius=.0025,
              rgb_mat=None,
              alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param axis_length:
    :param axis_radius:
    :param rgb_mat: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    endx = pos + rotmat[:, 0] * axis_length
    endy = pos + rotmat[:, 1] * axis_length
    endz = pos + rotmat[:, 2] * axis_length
    if rgb_mat is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgb_mat[:, 0]
        rgby = rgb_mat[:, 1]
        rgbz = rgb_mat[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # - 20201202 change it to ModelCollection
    # + 20230813 changing to ModelCollection seems unnecessary
    frame_nodepath = NodePath("frame")
    arrowx_trm = trm_factory.gen_arrow(spos=pos, epos=endx, stick_radius=axis_radius)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MAlpha)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trm_factory.gen_arrow(spos=pos, epos=endy, stick_radius=axis_radius)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MAlpha)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = trm_factory.gen_arrow(spos=pos, epos=endz, stick_radius=axis_radius)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MAlpha)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_2d_frame(pos=np.array([0, 0, 0]),
                 rotmat=np.eye(3),
                 axis_length=.1,
                 axis_radius=.0025,
                 rgb_mat=None,
                 alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param axis_length:
    :param axis_radius:
    :param rgb_mat: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20230913
    """
    endx = pos + rotmat[:, 0] * axis_length
    endy = pos + rotmat[:, 1] * axis_length
    if rgb_mat is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
    else:
        rgbx = rgb_mat[:, 0]
        rgby = rgb_mat[:, 1]
    if alpha is None:
        alphax = alphay = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
    else:
        alphax = alphay = alpha
    # - 20201202 change it to ModelCollection
    # + 20230813 changing to ModelCollection seems unnecessary
    frame_nodepath = NodePath("frame")
    arrowx_trm = trm_factory.gen_arrow(spos=pos, epos=endx, stick_radius=axis_radius)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MAlpha)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trm_factory.gen_arrow(spos=pos, epos=endy, stick_radius=axis_radius)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MAlpha)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_wireframe(vertices,
                  edges,
                  thickness=0.001,
                  rgba=np.array([0, 0, 0, 1])):
    """
    gen wireframe
    :param vertices: (n,3)
    :param edges: (n,2) indices to vertices
    :param thickness:
    :param rgba:
    :return: a geomtric model
    author: weiwei
    date: 20230815
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(*rgba)
    for line_seg in edges:
        ls.moveTo(*vertices(line_seg[0]))
        ls.drawTo(*vertices(line_seg[1]))
    # Create and return a node with the segments
    ls_pdndp = NodePath(ls.create())
    ls_pdndp.setTransparency(TransparencyAttrib.MDual)
    ls_pdndp.setLightOff()
    ls_sgm = StaticGeometricModel(initor=ls_pdndp)
    return ls_sgm


def gen_rgb_frame(pos=np.array([0, 0, 0]),
                  rotmat=np.eye(3),
                  axis_length=.1,
                  axis_radius=.0025,
                  alpha=None):
    """
    gen an axis for attaching, use red for x, blue for y, green for z
    this is a helper function to gen_frame
    :param pos:
    :param rotmat:
    :param axis_length:
    :param axis_radius:
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20230813
    """
    return gen_frame(pos=pos,
                     rotmat=rotmat,
                     axis_length=axis_length,
                     axis_radius=axis_radius,
                     rgb_mat=cst.rgb_mat,
                     alpha=alpha)


def gen_myc_frame(pos=np.array([0, 0, 0]),
                  rotmat=np.eye(3),
                  axis_length=.1,
                  axis_radius=.0025,
                  alpha=None):
    """
    gen an axis for attaching, use magne for x, yellow for y, cyan for z
    this is a helper function to gen_frame
    :param pos:
    :param rotmat:
    :param axis_length:
    :param axis_radius:
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    return gen_frame(pos=pos,
                     rotmat=rotmat,
                     axis_length=axis_length,
                     axis_radius=axis_radius,
                     rgb_mat=cst.myc_mat,
                     alpha=alpha)


def gen_dashed_frame(pos=np.array([0, 0, 0]),
                     rotmat=np.eye(3),
                     axis_length=.1,
                     axis_radius=.0025,
                     len_solid=None,
                     len_interval=None,
                     rgb_mat=None,
                     alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param axis_length:
    :param axis_radius:
    :param len_solid: axis_length of the solid section, 1*major_radius by default
    :param len_interval: axis_length of the empty section, 1.5*major_radius by default
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20200630osaka
    """
    endx = pos + rotmat[:, 0] * axis_length
    endy = pos + rotmat[:, 1] * axis_length
    endz = pos + rotmat[:, 2] * axis_length
    if rgb_mat is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
        rgbz = np.array([0, 0, 1])
    else:
        rgbx = rgb_mat[:, 0]
        rgby = rgb_mat[:, 1]
        rgbz = rgb_mat[:, 2]
    if alpha is None:
        alphax = alphay = alphaz = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
        alphaz = alpha[2]
    else:
        alphax = alphay = alphaz = alpha
    # - 20201202 change it toModelCollection
    # + 20230813 changing to ModelCollection seems unnecessary
    frame_nodepath = NodePath("dash_frame")
    arrowx_trm = trm_factory.gen_dasharrow(spos=pos, epos=endx, stick_radius=axis_radius, len_solid=len_solid,
                                           len_interval=len_interval)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trm_factory.gen_dasharrow(spos=pos, epos=endy, stick_radius=axis_radius, len_solid=len_solid,
                                           len_interval=len_interval)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowz_trm = trm_factory.gen_dasharrow(spos=pos, epos=endz, stick_radius=axis_radius, len_solid=len_solid,
                                           len_interval=len_interval)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgbz[0], rgbz[1], rgbz[2], alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_2d_dashed_frame(pos=np.array([0, 0, 0]),
                        rotmat=np.eye(3),
                        axis_length=.1,
                        axis_radius=.0025,
                        len_solid=None,
                        len_interval=None,
                        rgb_mat=None,
                        alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param axis_length:
    :param axis_radius:
    :param len_solid: axis_length of the solid section, 1*major_radius by default
    :param len_interval: axis_length of the empty section, 1.5*major_radius by default
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20200630osaka
    """
    endx = pos + rotmat[:, 0] * axis_length
    endy = pos + rotmat[:, 1] * axis_length
    if rgb_mat is None:
        rgbx = np.array([1, 0, 0])
        rgby = np.array([0, 1, 0])
    else:
        rgbx = rgb_mat[:, 0]
        rgby = rgb_mat[:, 1]
    if alpha is None:
        alphax = alphay = 1
    elif isinstance(alpha, np.ndarray):
        alphax = alpha[0]
        alphay = alpha[1]
    else:
        alphax = alphay = alpha
    # - 20201202 change it toModelCollection
    # + 20230813 changing to ModelCollection seems unnecessary
    frame_nodepath = NodePath("dash_frame")
    arrowx_trm = trm_factory.gen_dasharrow(spos=pos, epos=endx, stick_radius=axis_radius, len_solid=len_solid,
                                           len_interval=len_interval)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trm_factory.gen_dasharrow(spos=pos, epos=endy, stick_radius=axis_radius, len_solid=len_solid,
                                           len_interval=len_interval)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath)
    return frame_sgm


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=.5,
              center=np.array([0, 0, 0]),
              major_radius=.005,
              minor_radius=.00075,
              rgba=np.array([1, 0, 0, 1]),
              n_sec_major=24,
              n_sec_minor=8):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    torus_trm = trm_factory.gen_torus(axis=axis,
                                      starting_vector=starting_vector,
                                      portion=portion,
                                      center=center,
                                      major_radius=major_radius,
                                      minor_radius=minor_radius,
                                      n_sec_major=n_sec_major,
                                      n_sec_minor=n_sec_minor)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_rgba(rgba=rgba)
    return torus_sgm


def gen_dashtorus(axis=np.array([1, 0, 0]),
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  major_radius=0.1,
                  minor_radius=0.0025,
                  rgba=np.array([1, 0, 0, 1]),
                  len_solid=None,
                  len_interval=None,
                  n_sec_major=24,
                  n_sec_minor=8):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :return:
    author: weiwei
    date: 20200602
    """
    torus_trm = trm_factory.gen_dashtorus(axis=axis,
                                          portion=portion,
                                          center=center,
                                          major_radius=major_radius,
                                          minor_radius=minor_radius,
                                          len_solid=len_solid,
                                          len_interval=len_interval,
                                          n_sec_major=n_sec_major,
                                          n_sec_minor=n_sec_minor)
    torus_sgm = StaticGeometricModel(torus_trm)
    torus_sgm.set_rgba(rgba=rgba)
    return torus_sgm


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  major_radius=.05,
                  minor_radius=.0025,
                  rgba=np.array([1, 0, 0, 1]),
                  n_sec_major=24,
                  n_sec_minor=8,
                  end_type='single'):
    """
    :param axis: the circ arrow will rotate around this axis 1x3 nparray
    :param portion: 0.0~1.0
    :param center: the center position of the circ 1x3 nparray
    :param end_type: 'single' or 'double'
    :return:
    author: weiwei
    date: 20200602
    """
    circarrow_trm = trm_factory.gen_circarrow(axis=axis,
                                              starting_vector=starting_vector,
                                              portion=portion,
                                              center=center,
                                              major_radius=major_radius,
                                              minor_radius=minor_radius,
                                              n_sec_major=n_sec_major,
                                              n_sec_minor=n_sec_minor,
                                              end_type=end_type)
    circarrow_sgm = StaticGeometricModel(circarrow_trm)
    circarrow_sgm.set_rgba(rgba=rgba)
    return circarrow_sgm


def gen_pointcloud(points, rgba=np.array([0, 0, 0, .7]), point_size=.001):
    """
    do not use this raw function directly
    use environment.collisionmodel to call it
    gen objmnp
    :param points: nx3 nparray
    :param rgba: None; Specify for each point; Specify a unified color
    :return:
    """
    pcd_pdndp = da.pdgeomndp_from_v(points, rgba)
    pcd_pdndp.setRenderModeThickness(point_size * da.M_TO_PIXEL)
    pointcloud_sgm = StaticGeometricModel(pcd_pdndp)
    return pointcloud_sgm


def gen_submesh(vertices, faces, rgba=np.array([1, 0, 0, 1])):
    """
    :param vertices: np.array([[v00, v01, v02], [v10, v11, v12], ...]
    :param faces: np.array([[ti00, ti01, ti02], [ti10, ti11, ti12], ...]
    :param color: rgba
    :return:
    author: weiwei
    date: 20171219
    """
    # gen vert normals
    vertex_normals = np.zeros((len(vertices), 3))
    for fc in vertices:
        vert0 = vertices[fc[0], :]
        vert1 = vertices[fc[1], :]
        vert2 = vertices[fc[2], :]
        facenormal = np.cross(vert2 - vert1, vert0 - vert1)
        vertex_normals[fc[0], :] = vertex_normals[fc[0]] + facenormal
        vertex_normals[fc[1], :] = vertex_normals[fc[1]] + facenormal
        vertex_normals[fc[2], :] = vertex_normals[fc[2]] + facenormal
    for i in range(0, len(vertex_normals)):
        vertex_normals[i, :] = vertex_normals[i, :] / np.linalg.norm(vertex_normals[i, :])
    trm_mesh = trm_factory.trm_from_vvnf(vertices, vertex_normals, faces)
    submesh_sgm = StaticGeometricModel(trm_mesh)
    submesh_sgm.set_rgba(rgba=rgba)
    # geom = da.pdgeom_from_vvnf(vertices, vertex_normals, faces)
    # node = GeomNode('surface')
    # node.addGeom(geom)
    # surface_nodepath = NodePath('surface')
    # surface_nodepath.attachNewNode(node)
    # surface_nodepath.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    # surface_nodepath.setTransparency(TransparencyAttrib.MDual)
    # surface_nodepath.setTwoSided(True)
    # surface_sgm = StaticGeometricModel(surface_nodepath)
    return submesh_sgm


def gen_polygon(verts, thickness=0.002, rgba=np.array([0, 0, 0, .7])):
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


def gen_frame_box(extent=np.array([.02, .02, .02]),
                  homomat=np.eye(4),
                  rgba=np.array([0, 0, 0, 1]),
                  thickness=.001):
    """
    draw a 3D box, only show edges
    :param extent:
    :param homomat:
    :return:
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
    center_pos = homomat[:3, 3]
    x_axis = homomat[:3, 0]
    y_axis = homomat[:3, 1]
    z_axis = homomat[:3, 2]
    x_min, x_max = -x_axis * extent[0] / 2, x_axis * extent[0] / 2
    y_min, y_max = -y_axis * extent[1] / 2, y_axis * extent[1] / 2
    z_min, z_max = -z_axis * extent[2] / 2, z_axis * extent[2] / 2
    # max, max, max
    print(center_pos + np.array([x_max, y_max, z_max]))
    ls.moveTo(da.npvec3_to_pdvec3(center_pos + x_max + y_max + z_max))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_max + y_max + z_min))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_max + y_min + z_min))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_max + y_min + z_max))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_max + y_max + z_max))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_max + z_max))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_min + z_max))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_min + z_min))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_max + z_min))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_max + z_max))
    ls.moveTo(da.npvec3_to_pdvec3(center_pos + x_max + y_max + z_min))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_max + z_min))
    ls.moveTo(da.npvec3_to_pdvec3(center_pos + x_max + y_min + z_min))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_min + z_min))
    ls.moveTo(da.npvec3_to_pdvec3(center_pos + x_max + y_min + z_max))
    ls.drawTo(da.npvec3_to_pdvec3(center_pos + x_min + y_min + z_max))
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp)
    return ls_sgm


def gen_surface(surface_callback, rng, granularity=.01):
    surface_trm = trm_factory.gen_surface(surface_callback, rng, granularity)
    surface_gm = GeometricModel(surface_trm, toggle_twosided=True)
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
    bunnygm.show_local_frame()
    rotmat = rm.rotmat_from_axangle(np.array([1, 0, 0]), math.pi / 2.0)
    bunnygm.rotmat = rotmat

    bunnygm1 = bunnygm.copy()
    bunnygm1.set_rgba([0.7, 0, 0.7, 1.0])
    bunnygm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnygm1.pos = np.array([0, .01, 0])
    bunnygm1.rotmat = rotmat

    bunnygm2 = bunnygm1.copy()
    bunnygm2.set_rgba([0, 0.7, 0.7, 1.0])
    bunnygm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnygm2.pos = np.array([0, .2, 0])
    bunnygm2.rotmat = rotmat
    bunnygm2.set_scale([2, 1, 3])

    bunnygmpoints = bunnygm.sample_surface()
    bunnygm1points = bunnygm1.sample_surface()
    bunnygm2points = bunnygm2.sample_surface()
    # bpgm = GeometricModel(bunnygmpoints)
    # bpgm1 = GeometricModel(bunnygm1points)
    # bpgm2 = GeometricModel(bunnygm2points)
    # bpgm.attach_to(base)
    # bpgm.set_scale([2, 1, 3])
    # bpgm.set_point_size(.01)
    # bpgm1.attach_to(base)
    # bpgm2.attach_to(base)
    bgm_pcd = gen_pointcloud(bunnygmpoints)
    bgm_pcd.attach_to(base)

    lsgm = gen_linesegs([[np.array([.1, 0, .01]), np.array([.01, 0, .01])],
                         [np.array([.01, 0, .01]), np.array([.1, 0, .1])],
                         [np.array([.1, 0, .1]), np.array([.1, 0, .01])]])
    lsgm.attach_to(base)

    gen_circarrow(major_radius=.1, portion=.8).attach_to(base)
    gen_dashed_arrow(spos=np.array([0, 0, 0]), epos=np.array([0, 0, 2])).attach_to(base)
    gen_dashed_frame(pos=np.array([0, 0, 0]), rotmat=np.eye(3)).attach_to(base)
    axmat = rm.rotmat_from_axangle([1, 1, 1], math.pi / 4)
    gen_frame(rotmat=axmat).attach_to(base)
    axmat[:, 0] = .1 * axmat[:, 0]
    axmat[:, 1] = .07 * axmat[:, 1]
    axmat[:, 2] = .3 * axmat[:, 2]
    gen_ellipsoid(pos=np.array([0, 0, 0]), axes_mat=axmat).attach_to(base)
    print(rm.unit_vector(np.array([0, 0, 0])))

    pos = np.array([.3, 0, 0])
    rotmat = rm.rotmat_from_euler(math.pi / 6, 0, 0)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    gen_frame_box([.1, .2, .3], homomat).attach_to(base)

    base.run()
