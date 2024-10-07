import copy
import numpy as np
from panda3d.core import NodePath, LineSegs, TransparencyAttrib
from direct.showbase.ShowBase import ShowBase
import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
import wrs.modeling.model_collection as mmc
import wrs.basis.trimesh_factory as trm_factory

try:
    import open3d as o3d
except:
    o3d = None


# ==================================
# definition of StaticGeometricModel
# ==================================

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
                 toggle_twosided=False,
                 rgb=rm.const.tab20_list[0],
                 alpha=1):
        """
        :param initor: path end_type defined by os.path or trimesh or pdndp
        :param toggle_transparency
        :param name
        """
        if isinstance(initor, StaticGeometricModel):
            if initor.trm_mesh is not None:
                self.__init__(initor=initor.trm_mesh, name=initor.name,
                              toggle_transparency=initor.pdndp.getTransparency(),
                              toggle_twosided=initor.pdndp.getTwoSided(),
                              rgb=initor.rgb,
                              alpha=initor.alpha)
            else:
                self.__init__(initor=initor.pdndp, name=initor.name,
                              toggle_transparency=initor.pdndp.getTransparency(),
                              toggle_twosided=initor.pdndp.getTwoSided(),
                              rgb=initor.rgb,
                              alpha=initor.alpha)
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
            elif isinstance(initor, NodePath):  # keeping this one to allow efficient frame representations, 20240311
                self._file_path = None
                self._trm_mesh = None
                pdndp_core = initor
                pdndp_core.reparentTo(self._pdndp)
            elif initor is None:  # empty model elevated to here to allow not installing open3d
                self._file_path = None
                self._trm_mesh = None
                pdndp_core = NodePath("pdndp_core")
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, o3d.geometry.PointCloud):  # TODO should pointcloud be pdndp or pdnp_raw
                self._file_path = None
                self._trm_mesh = da.trm.Trimesh(np.asarray(initor.points))
                pdndp_core = da.pdgeomndp_from_v(self._trm_mesh.vertices, name='pdndp_core')
                pdndp_core.reparentTo(self._pdndp)
            elif isinstance(initor, o3d.geometry.TriangleMesh):
                self._file_path = None
                self._trm_mesh = da.trm.Trimesh(vertices=initor.vertices, faces=initor.triangles,
                                                face_normals=initor.triangle_normals)
                pdndp_core = da.trimesh_to_nodepath(self._trm_mesh, name='pdndp_core')
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
            if rgb is not None:
                self._pdndp.setColor(rgb[0], rgb[1], rgb[2], alpha)
            self._pdndp.setMaterialOff()
            self._pdndp.setShaderAuto()
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

    @property
    def rgba(self):
        return da.pdvec4_to_npvec4(self._pdndp.getColor())

    @rgba.setter
    def rgba(self, rgba):
        if rgba is None:
            self._pdndp.clearColor()
        else:
            self._pdndp.setColor(*rgba)

    @property
    def rgb(self):
        return self.rgba[:3]

    @rgb.setter
    def rgb(self, rgb):
        self._pdndp.setColor(rgb[0], rgb[1], rgb[2], self.alpha)

    @property
    def alpha(self):
        return self._pdndp.getColor()[3]

    @alpha.setter
    def alpha(self, alpha):
        rgba = self._pdndp.getColor()
        self._pdndp.setColor(rgba[0], rgba[1], rgba[2], alpha)

    def is_empty(self):
        if self._trm_mesh is None:
            return True
        else:
            return False

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

    def attach_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._pdndp.reparentTo(target.render)
        elif isinstance(target, StaticGeometricModel):  # prepared for decorations like local frames
            self._pdndp.reparentTo(target.pdndp)
        elif isinstance(target, mmc.ModelCollection):
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


# ============================
# definition of GeometricModel
# ============================

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
                 toggle_twosided=False,
                 rgb=rm.const.tab20_list[0],
                 alpha=1):
        """
        :param initor: path end_type defined by os.path or trimesh or pdndp
        """
        if isinstance(initor, GeometricModel):
            if initor.trm_mesh is not None:
                super().__init__(initor=initor.trm_mesh,
                                 name=name,
                                 toggle_transparency=initor.pdndp.getTransparency(),
                                 toggle_twosided=initor.pdndp.getTwoSided())
            else:
                super().__init__(initor=initor.pdndp,
                                 name=name,
                                 toggle_transparency=initor.pdndp.getTransparency(),
                                 toggle_twosided=initor.pdndp.getTwoSided())
            self._pos = initor.pos
            self._rotmat = initor.rotmat
            self._is_pdndp_pose_delayed = True
            self.pdndp.setColor(initor.pdndp.getColor())
            # self.pdndp_core.setShaderAuto()
        else:
            super().__init__(initor=initor,
                             name=name,
                             toggle_transparency=toggle_transparency,
                             toggle_twosided=toggle_twosided,
                             rgb=rgb,
                             alpha=alpha)
            self._pos = np.zeros(3)
            self._rotmat = np.eye(3)
            self._is_pdndp_pose_delayed = True
            # self.pdndp_core.setShaderAuto()

    @staticmethod
    def delay_pdndp_pose_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_pdndp_pose_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_pdndp_pose_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_pdndp_pose_delayed:
                self._pdndp.setPosQuat(da.npvec3_to_pdvec3(self._pos), da.npmat3_to_pdquat(self._rotmat))
                self._is_pdndp_pose_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    @property
    @update_pdndp_pose_decorator
    def pdndp(self):
        return self._pdndp

    @property
    def pos(self):
        return self._pos

    @pos.setter
    @delay_pdndp_pose_decorator
    def pos(self, pos: np.ndarray):
        self._pos = pos

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    @delay_pdndp_pose_decorator
    def rotmat(self, rotmat: np.ndarray):
        self._rotmat = rotmat

    @property
    def homomat(self):
        homomat = np.eye(4)
        homomat[:3, :3] = self._rotmat
        homomat[:3, 3] = self._pos
        return homomat

    @homomat.setter
    @delay_pdndp_pose_decorator
    def homomat(self, homomat: np.ndarray):
        self._pos = homomat[:3, 3]
        self._rotmat = homomat[:3, :3]

    @property
    def pose(self):
        """
        a pose is defined as a tuple with the first element being npvec3, the second element beging npmat3
        :return:
        author: weiwei
        date: 20231123
        """
        return (self._pos, self._rotmat)

    @pose.setter
    @delay_pdndp_pose_decorator
    def pose(self, pose):
        """
        :param pose: tuple or list containing an npvec3 and an npmat3
        :return:
        """
        self._pos = pose[0]
        self._rotmat = pose[1]

    def set_transparency(self, attribute):
        return self._pdndp.setTransparency(attribute)

    @update_pdndp_pose_decorator
    def attach_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._pdndp.reparentTo(target.render)
        elif isinstance(target, StaticGeometricModel):  # prepared for decorations like local frames
            self._pdndp.reparentTo(target.pdndp)
        elif isinstance(target, mmc.ModelCollection):
            target.add_gm(self)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, ModelCollection!")

    @update_pdndp_pose_decorator
    def attach_copy_to(self, target):
        """
        attach a copy of pdndp to target
        :param target:
        :return:
        """
        pdndp = copy.deepcopy(self._pdndp)
        if isinstance(target, ShowBase):
            pdndp.reparentTo(target.render)
        elif isinstance(target, StaticGeometricModel):  # prepared for decorations like local frames
            pdndp.reparentTo(target.pdndp)
        elif isinstance(target, NodePath):
            pdndp.reparentTo(target)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, NodePath!")
        return pdndp

    def detach(self):  # TODO detach from?
        self._pdndp.detachNode()

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
        gmodel = GeometricModel(self)
        gmodel.pos = self.pos
        gmodel.rotmat = self.rotmat
        return gmodel


# ======================================================
# helper functions for creating various geometric models
# ======================================================

def gen_linesegs(linesegs,
                 thickness=0.001,
                 rgb=np.array([0, 0, 0]),
                 alpha=1):
    """
    gen linsegs -- non-continuous segs are allowed
    :param linesegs: nx2x3 nparray, defined in local 0 frame
    :param rgb:
    :param alpha:
    :param thickness:
    :param refpos, refrot: the local coordinate frame where the pnti in the linsegs are defined
    :return: a geomtric model
    author: weiwei
    date: 20161216, 20201116, 20230812
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(rgb[0], rgb[1], rgb[2], alpha)
    for p0_p1_tuple in linesegs:
        ls.moveTo(*p0_p1_tuple[0])
        ls.drawTo(*p0_p1_tuple[1])
    # Create and return a node with the segments
    ls_pdndp = NodePath(ls.create())
    ls_pdndp.setTransparency(TransparencyAttrib.MDual)
    ls_pdndp.setLightOff()
    ls_sgm = StaticGeometricModel(initor=ls_pdndp, rgb=None)
    return ls_sgm


def gen_sphere(pos=np.array([0, 0, 0]),
               radius=0.0015,
               rgb=np.array([1, 0, 0]),
               alpha=1,
               ico_level=2):
    """
    :param pos:
    :param radius:
    :param rgb:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_trm = trm_factory.gen_sphere(pos=pos, radius=radius, ico_level=ico_level)
    sphere_sgm = StaticGeometricModel(initor=sphere_trm, rgb=rgb, alpha=alpha)
    return sphere_sgm


def gen_ellipsoid(pos=np.array([0, 0, 0]),
                  axes_mat=np.eye(3),
                  rgb=np.array([1, 1, 0]),
                  alpha=.3):
    """
    :param pos:
    :param axes_mat: 3x3 mat, each column is an axis of the ellipse
    :param rgb:
    :param alpha
    :return:
    author: weiwei
    date: 20200701osaka
    """
    ellipsoid_trm = trm_factory.gen_ellipsoid(pos=pos, axmat=axes_mat)
    ellipsoid_sgm = StaticGeometricModel(initor=ellipsoid_trm, rgb=rgb, alpha=alpha)
    return ellipsoid_sgm


def gen_stick(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              radius=.0025,
              rgb=np.array([1, 0, 0]),
              alpha=1,
              type="rect",
              n_sec=18):
    """
    :param spos:
    :param epos:
    :param radius:
    :param rgb:
    :param type: rect or round
    :param n_sec:
    :return:
    author: weiwei
    date: 20191229osaka
    """
    stick_trm = trm_factory.gen_stick(spos=spos, epos=epos, radius=radius, type=type, n_sec=n_sec)
    stick_sgm = StaticGeometricModel(initor=stick_trm, rgb=rgb, alpha=alpha)
    return stick_sgm


def gen_dashed_stick(spos=np.array([0, 0, 0]),
                     epos=np.array([.1, 0, 0]),
                     radius=.0025,
                     rgb=np.array([1, 0, 0]),
                     alpha=1,
                     len_solid=None,
                     len_interval=None,
                     type="rect",
                     n_sec=18):
    """
    :param spos:
    :param epos:
    :param radius:
    :param len_solid: ax_length of the solid section, 1*major_radius by default
    :param len_interval: ax_length of the interval between two solids, 1.5*major_radius by default
    :param rgb:
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
    dashstick_sgm = StaticGeometricModel(initor=dashstick_trm, rgb=rgb, alpha=alpha)
    return dashstick_sgm


def gen_box(xyz_lengths=np.array([1, 1, 1]),
            pos=np.zeros(3),
            rotmat=np.eye(3),
            rgb=rm.const.steel_gray,
            alpha=1):
    """
    :param xyz_lengths:
    :param pos:
    :param rotmat:
    :param rgb:
    :param alpha:
    :return:
    author: weiwei
    date: 20191229osaka, 20230830, 20240616osaka
    """
    box_trm = trm_factory.gen_box(xyz_lengths=xyz_lengths, pos=pos, rotmat=rotmat)
    box_sgm = StaticGeometricModel(initor=box_trm, rgb=rgb, alpha=alpha)
    return box_sgm


def gen_frustrum(bottom_xy_lengths=np.array([0.02, 0.02]), top_xy_lengths=np.array([0.04, 0.04]),
                 height=0.01, pos=np.zeros(3), rotmat=np.eye(3), rgb=np.array([1, 0, 0]), alpha=1):
    """
    Draw a 3D frustum
    :param bottom_xy_lengths: XYZ lengths of the bottom rectangle
    :param top_xy_lengths: XYZ lengths of the top rectangle
    :param height: Height of the frustum
    :param pos: Position of the frustum center
    :param rotmat: Rotation matrix for the frustum orientation
    :param rgb: Color of the frustum
    :param alpha:
    :return: A NodePath with the frustum geometry
    """
    frustrum_trm = trm_factory.gen_frustrum(bottom_xy_lengths=bottom_xy_lengths,
                                            top_xy_lengths=top_xy_lengths,
                                            height=height, pos=pos, rotmat=rotmat)
    ls_sgm = StaticGeometricModel(initor=frustrum_trm, rgb=rgb, alpha=alpha)
    return ls_sgm


def gen_dumbbell(spos=np.array([0, 0, 0]),
                 epos=np.array([.1, 0, 0]),
                 rgb=np.array([1, 0, 0]),
                 alpha=1,
                 stick_radius=.0025,
                 n_sec=18,
                 sphere_radius=None,
                 sphere_ico_level=2):
    """
    :param sphere_radius:
    :param spos:
    :param epos:
    :param stick_radius:
    :param rgb:
    :param alpha
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
    dumbbell_sgm = StaticGeometricModel(dumbbell_trm, rgb=rgb, alpha=alpha)
    return dumbbell_sgm


def gen_cone(spos=np.array([0, 0, 0]),
             epos=np.array([0.1, 0, 0]),
             rgb=np.array([.7, .7, .7]),
             alpha=1,
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
    cone_sgm = GeometricModel(cone_trm, rgb=rgb, alpha=alpha)
    return cone_sgm


def gen_arrow(spos=np.array([0, 0, 0]),
              epos=np.array([.1, 0, 0]),
              rgb=np.array([1, 0, 0]),
              alpha=1,
              stick_radius=.0025,
              stick_type="rect"):
    """
    :param spos:
    :param epos:
    :param stick_radius:
    :param rgb:
    :param alpha
    :return:
    author: weiwei
    date: 20200115osaka
    """
    arrow_trm = trm_factory.gen_arrow(spos=spos, epos=epos, stick_radius=stick_radius, stick_type=stick_type)
    arrow_sgm = StaticGeometricModel(arrow_trm, rgb=rgb, alpha=alpha)
    return arrow_sgm


def gen_dashed_arrow(spos=np.array([0, 0, 0]),
                     epos=np.array([.1, 0, 0]),
                     rgb=np.array([1, 0, 0]),
                     alpha=1,
                     stick_radius=.0025,
                     len_solid=None,
                     len_interval=None,
                     type="rect"):
    """
    :param spos:
    :param epos:
    :param stick_radius:
    :param len_solid: ax_length of the solid section, 1*major_radius by default
    :param len_interval: ax_length of the empty section, 1.5*major_radius by default
    :param rgb:
    :param alpha
    :return:
    author: weiwei
    date: 20200625osaka
    """
    dasharrow_trm = trm_factory.gen_dashed_arrow(spos=spos,
                                                 epos=epos,
                                                 len_solid=len_solid,
                                                 len_interval=len_interval,
                                                 stick_radius=stick_radius,
                                                 stick_type=type)
    dasharrow_sgm = StaticGeometricModel(dasharrow_trm, rgb=rgb, alpha=alpha)
    return dasharrow_sgm


def gen_frame(pos=np.array([0, 0, 0]),
              rotmat=np.eye(3),
              ax_length=.1,
              ax_radius=.0025,
              rgb_mat=None,
              alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param ax_length:
    :param ax_radius:
    :param rgb_mat: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    end_pos = (pos + rotmat.T * ax_length).T
    if rgb_mat is None:
        rgb_mat = rm.const.rgb_mat
    if alpha is None:
        alpha = [1] * 3
    elif not isinstance(alpha, np.ndarray):
        alpha = [alpha] * 3
    frame_nodepath = NodePath("frame")
    arrowx_trm = trm_factory.gen_arrow(spos=pos, epos=end_pos[:, 0], stick_radius=ax_radius)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgb_mat[:, 0][0], rgb_mat[:, 0][1], rgb_mat[:, 0][2], alpha[0])
    arrowy_trm = trm_factory.gen_arrow(spos=pos, epos=end_pos[:, 1], stick_radius=ax_radius)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgb_mat[:, 1][0], rgb_mat[:, 1][1], rgb_mat[:, 1][2], alpha[1])
    arrowz_trm = trm_factory.gen_arrow(spos=pos, epos=end_pos[:, 2], stick_radius=ax_radius)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(rgb_mat[:, 2][0], rgb_mat[:, 2][1], rgb_mat[:, 2][2], alpha[2])
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath, rgb=None)
    return frame_sgm


def gen_2d_frame(pos=np.array([0, 0, 0]),
                 rotmat=np.eye(3),
                 ax_length=.1,
                 ax_radius=.0025,
                 rgb_mat=None,
                 alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param ax_length:
    :param ax_radius:
    :param rgb_mat: each column indicates the color of each base
    :param plotname:
    :return:
    author: weiwei
    date: 20230913
    """
    endx = pos + rotmat[:, 0] * ax_length
    endy = pos + rotmat[:, 1] * ax_length
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
    frame_nodepath = NodePath("frame")
    arrowx_trm = trm_factory.gen_arrow(spos=pos, epos=endx, stick_radius=ax_radius)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trm_factory.gen_arrow(spos=pos, epos=endy, stick_radius=ax_radius)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath, rgb=None)
    return frame_sgm


def gen_wireframe(vertices,
                  edges,
                  thickness=0.001,
                  rgb=np.array([0, 0, 0]),
                  alpha=1):
    """
    gen wireframe
    :param vertices: (n,3)
    :param edges: (n,2) indices to vertices
    :param thickness:
    :param rgb:
    :param alpha:
    :return: a geomtric model
    author: weiwei
    date: 20230815
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(*rgb, alpha)
    for line_seg in edges:
        ls.moveTo(*vertices(line_seg[0]))
        ls.drawTo(*vertices(line_seg[1]))
    # Create and return a node with the segments
    ls_pdndp = NodePath(ls.create())
    ls_pdndp.setTransparency(TransparencyAttrib.MDual)
    ls_pdndp.setLightOff()
    ls_sgm = StaticGeometricModel(initor=ls_pdndp, rgb=None)
    return ls_sgm


def gen_rgb_frame(pos=np.array([0, 0, 0]),
                  rotmat=np.eye(3),
                  ax_length=.1,
                  ax_radius=.0025,
                  alpha=None):
    """
    gen an axis for attaching, use red for x, blue for y, green for z
    this is a helper function to gen_frame
    :param pos:
    :param rotmat:
    :param ax_length:
    :param ax_radius:
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20230813
    """
    return gen_frame(pos=pos,
                     rotmat=rotmat,
                     ax_length=ax_length,
                     ax_radius=ax_radius,
                     rgb_mat=rm.const.rgb_mat,
                     alpha=alpha)


def gen_myc_frame(pos=np.array([0, 0, 0]),
                  rotmat=np.eye(3),
                  ax_length=.1,
                  ax_radius=.0025,
                  alpha=None):
    """
    gen an axis for attaching, use magne for x, yellow for y, cyan for z
    this is a helper function to gen_frame
    :param pos:
    :param rotmat:
    :param ax_length:
    :param ax_radius:
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    return gen_frame(pos=pos,
                     rotmat=rotmat,
                     ax_length=ax_length,
                     ax_radius=ax_radius,
                     rgb_mat=rm.const.myc_mat,
                     alpha=alpha)


def gen_dashed_frame(pos=np.array([0, 0, 0]),
                     rotmat=np.eye(3),
                     ax_length=.1,
                     ax_radius=.0025,
                     len_solid=None,
                     len_interval=None,
                     rgb_mat=None,
                     alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param ax_length:
    :param ax_radius:
    :param len_solid: ax_length of the solid section, 1*major_radius by default
    :param len_interval: ax_length of the empty section, 1.5*major_radius by default
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20200630osaka
    """
    endx = pos + rotmat[:, 0] * ax_length
    endy = pos + rotmat[:, 1] * ax_length
    endz = pos + rotmat[:, 2] * ax_length
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
    arrowx_trm = trm_factory.gen_dashed_arrow(spos=pos, epos=endx, stick_radius=ax_radius, len_solid=len_solid,
                                              len_interval=len_interval)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(*rgbx, alphax)
    arrowy_trm = trm_factory.gen_dashed_arrow(spos=pos, epos=endy, stick_radius=ax_radius, len_solid=len_solid,
                                              len_interval=len_interval)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(*rgby, alphay)
    arrowz_trm = trm_factory.gen_dashed_arrow(spos=pos, epos=endz, stick_radius=ax_radius, len_solid=len_solid,
                                              len_interval=len_interval)
    arrowz_nodepath = da.trimesh_to_nodepath(arrowz_trm)
    arrowz_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowz_nodepath.setColor(*rgbz, alphaz)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    arrowz_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath, rgb=None)
    return frame_sgm


def gen_2d_dashed_frame(pos=np.array([0, 0, 0]),
                        rotmat=np.eye(3),
                        ax_length=.1,
                        ax_radius=.0025,
                        len_solid=None,
                        len_interval=None,
                        rgb_mat=None,
                        alpha=None):
    """
    gen an axis for attaching
    :param pos:
    :param rotmat:
    :param ax_length:
    :param ax_radius:
    :param len_solid: ax_length of the solid section, 1*major_radius by default
    :param len_interval: ax_length of the empty section, 1.5*major_radius by default
    :param rgb_mat: each column indicates the color of each base
    :return:
    author: weiwei
    date: 20200630osaka
    """
    endx = pos + rotmat[:, 0] * ax_length
    endy = pos + rotmat[:, 1] * ax_length
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
    arrowx_trm = trm_factory.gen_dashed_arrow(spos=pos, epos=endx, stick_radius=ax_radius, len_solid=len_solid,
                                              len_interval=len_interval)
    arrowx_nodepath = da.trimesh_to_nodepath(arrowx_trm)
    arrowx_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowx_nodepath.setColor(rgbx[0], rgbx[1], rgbx[2], alphax)
    arrowy_trm = trm_factory.gen_dashed_arrow(spos=pos, epos=endy, stick_radius=ax_radius, len_solid=len_solid,
                                              len_interval=len_interval)
    arrowy_nodepath = da.trimesh_to_nodepath(arrowy_trm)
    arrowy_nodepath.setTransparency(TransparencyAttrib.MDual)
    arrowy_nodepath.setColor(rgby[0], rgby[1], rgby[2], alphay)
    arrowx_nodepath.reparentTo(frame_nodepath)
    arrowy_nodepath.reparentTo(frame_nodepath)
    frame_sgm = StaticGeometricModel(frame_nodepath, rgb=None)
    return frame_sgm


def gen_torus(axis=np.array([1, 0, 0]),
              starting_vector=None,
              portion=1,
              center=np.array([0, 0, 0]),
              major_radius=.2,
              minor_radius=.0015,
              rgb=np.array([1, 0, 0]),
              alpha=1,
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
    torus_sgm = StaticGeometricModel(torus_trm, rgb=rgb, alpha=alpha)
    return torus_sgm


def gen_dashed_torus(axis=np.array([1, 0, 0]),
                     portion=1,
                     center=np.array([0, 0, 0]),
                     major_radius=0.1,
                     minor_radius=0.0025,
                     rgb=np.array([1, 0, 0]),
                     alpha=1,
                     len_solid=None,
                     len_interval=None,
                     n_sec_major=64,
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
    torus_sgm = StaticGeometricModel(torus_trm, rgb=rgb, alpha=alpha)
    return torus_sgm


def gen_circarrow(axis=np.array([1, 0, 0]),
                  starting_vector=None,
                  portion=.5,
                  center=np.array([0, 0, 0]),
                  major_radius=.05,
                  minor_radius=.0025,
                  rgb=np.array([1, 0, 0]),
                  alpha=1,
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
    circarrow_sgm = StaticGeometricModel(circarrow_trm, rgb=rgb, alpha=alpha)
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
    pointcloud_sgm = StaticGeometricModel(pcd_pdndp, rgb=None)
    return pointcloud_sgm


def gen_submesh(vertices, faces, rgb=np.array([1, 0, 0]), alpha=1):
    """
    :param vertices: np.array([[v00, v01, v02], [v10, v11, v12], ...]
    :param faces: np.array([[ti00, ti01, ti02], [ti10, ti11, ti12], ...]
    :param color: rgb
    :param alpha
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
        face_normal = np.cross(vert2 - vert1, vert0 - vert1)
        vertex_normals[fc[0], :] = vertex_normals[fc[0]] + face_normal
        vertex_normals[fc[1], :] = vertex_normals[fc[1]] + face_normal
        vertex_normals[fc[2], :] = vertex_normals[fc[2]] + face_normal
    for i in range(0, len(vertex_normals)):
        vertex_normals[i, :] = vertex_normals[i, :] / np.linalg.norm(vertex_normals[i, :])
    trm_mesh = trm_factory.trm_from_vvnf(vertices, vertex_normals, faces)
    submesh_sgm = StaticGeometricModel(trm_mesh, rgb=rgb, alpha=alpha)
    return submesh_sgm


def gen_polygon(verts, thickness=0.002, rgb=np.array([0, 0, 0]), alpha=.7):
    """
    gen objmnp
    :param obj_path:
    :return:a
    author: weiwei
    date: 20201115
    """
    segs = LineSegs()
    segs.setThickness(thickness)
    segs.setColor(*rgb, alpha)
    for i in range(len(verts) - 1):
        segs.moveTo(verts[i][0], verts[i][1], verts[i][2])
        segs.drawTo(verts[i + 1][0], verts[i + 1][1], verts[i + 1][2])
    polygon_nodepath = NodePath('polygons')
    polygon_nodepath.attachNewNode(segs.create())
    polygon_nodepath.setTransparency(TransparencyAttrib.MDual)
    polygon_sgm = StaticGeometricModel(polygon_nodepath, rgb=None)
    return polygon_sgm


def gen_frame_box(xyz_lengths=np.array([.02, .02, .02]),
                  pos=np.zeros(3),
                  rotmat=np.eye(3),
                  rgb=np.array([0, 0, 0]),
                  alpha=1,
                  thickness=.001):
    """
    draw a 3d frame box
    :param xyz_lengths:
    :param pos:
    :param rotmat:
    :param rgb:
    :param alpha:
    :param thickness:
    :return:
    """
    # Create a set of line segments
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(rgb[0], rgb[1], rgb[2], alpha)
    center_pos = pos
    x_axis = rotmat[:, 0]
    y_axis = rotmat[:, 1]
    z_axis = rotmat[:, 2]
    x_min, x_max = -x_axis * xyz_lengths[0] / 2, x_axis * xyz_lengths[0] / 2
    y_min, y_max = -y_axis * xyz_lengths[1] / 2, y_axis * xyz_lengths[1] / 2
    z_min, z_max = -z_axis * xyz_lengths[2] / 2, z_axis * xyz_lengths[2] / 2
    # max, max, max
    print(center_pos + np.array([x_max, y_max, z_max]))
    ls.moveTo(*(center_pos + x_max + y_max + z_max))
    ls.drawTo(*(center_pos + x_max + y_max + z_min))
    ls.drawTo(*(center_pos + x_max + y_min + z_min))
    ls.drawTo(*(center_pos + x_max + y_min + z_max))
    ls.drawTo(*(center_pos + x_max + y_max + z_max))
    ls.drawTo(*(center_pos + x_min + y_max + z_max))
    ls.drawTo(*(center_pos + x_min + y_min + z_max))
    ls.drawTo(*(center_pos + x_min + y_min + z_min))
    ls.drawTo(*(center_pos + x_min + y_max + z_min))
    ls.drawTo(*(center_pos + x_min + y_max + z_max))
    ls.moveTo(*(center_pos + x_max + y_max + z_min))
    ls.drawTo(*(center_pos + x_min + y_max + z_min))
    ls.moveTo(*(center_pos + x_max + y_min + z_min))
    ls.drawTo(*(center_pos + x_min + y_min + z_min))
    ls.moveTo(*(center_pos + x_max + y_min + z_max))
    ls.drawTo(*(center_pos + x_min + y_min + z_max))
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp, rgb=None)
    return ls_sgm


def gen_frame_cylinder(radius=0.02, height=0.01, num_sides=8, pos=np.zeros(3), rotmat=np.eye(3),
                       rgb=np.array([0, 0, 0]), alpha=1, thickness=.001):
    """
    Draw a 3D cylinder using LineSegs
    :param radius: Radius of the cylinder
    :param height: Height of the cylinder
    :param num_sides: Number of sides for the cylindrical approximation
    :param pos: Position of the cylinder center
    :param rotmat: Rotation matrix for the cylinder orientation
    :param rgb: Color of the cylinder
    :param alpha
    :param thickness: Thickness of the lines
    :return: A NodePath with the cylinder geometry
    """
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(*rgb, alpha)
    angle_step = 2 * np.pi / num_sides
    half_height = height / 2
    top_circle = []
    bottom_circle = []
    for i in range(num_sides + 1):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        top_vertex = np.dot(rotmat, [x, y, half_height]) + pos
        bottom_vertex = np.dot(rotmat, [x, y, -half_height]) + pos
        top_circle.append(top_vertex)
        bottom_circle.append(bottom_vertex)
        # Draw the vertical lines
        if i > 0:
            ls.moveTo(*top_circle[i - 1])
            ls.drawTo(*top_circle[i])
            ls.moveTo(*bottom_circle[i - 1])
            ls.drawTo(*bottom_circle[i])
            ls.moveTo(*top_circle[i - 1])
            ls.drawTo(*bottom_circle[i - 1])
    # Draw the last set of vertical lines to close the cylinder
    ls.moveTo(*top_circle[-1])
    ls.drawTo(*top_circle[0])
    ls.moveTo(*bottom_circle[-1])
    ls.drawTo(*bottom_circle[0])
    ls.moveTo(*top_circle[-1])
    ls.drawTo(*bottom_circle[-1])
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp, rgb=None)
    return ls_sgm


def gen_frame_frustum(bottom_xy_lengths=np.array([0.02, 0.02]), top_xy_lengths=np.array([0.04, 0.04]),
                      height=0.01, pos=np.zeros(3), rotmat=np.eye(3), rgb=np.array([0, 0, 0]),
                      alpha=1, thickness=0.001):
    """
    Draw a 3D frustum using LineSegs
    :param bottom_xy_lengths: XYZ lengths of the bottom rectangle
    :param top_xy_lengths: XYZ lengths of the top rectangle
    :param height: Height of the frustum
    :param pos: Position of the frustum center
    :param rotmat: Rotation matrix for the frustum orientation
    :param rgb: Color of the frustum
    :param alpha
    :param thickness: Thickness of the lines
    :return: A NodePath with the frustum geometry
    """
    ls = LineSegs()
    ls.setThickness(thickness * da.M_TO_PIXEL)
    ls.setColor(*rgb, alpha)
    # Calculate vertices for the bottom and top rectangles
    bottom_offsets = [
        [-bottom_xy_lengths[0] / 2, -bottom_xy_lengths[1] / 2, 0],
        [bottom_xy_lengths[0] / 2, -bottom_xy_lengths[1] / 2, 0],
        [bottom_xy_lengths[0] / 2, bottom_xy_lengths[1] / 2, 0],
        [-bottom_xy_lengths[0] / 2, bottom_xy_lengths[1] / 2, 0]
    ]
    top_offsets = [
        [-top_xy_lengths[0] / 2, -top_xy_lengths[1] / 2, height],
        [top_xy_lengths[0] / 2, -top_xy_lengths[1] / 2, height],
        [top_xy_lengths[0] / 2, top_xy_lengths[1] / 2, height],
        [-top_xy_lengths[0] / 2, top_xy_lengths[1] / 2, height]
    ]
    bottom_vertices = [np.dot(rotmat, offset) + pos for offset in bottom_offsets]
    top_vertices = [np.dot(rotmat, offset) + pos for offset in top_offsets]
    # Draw bottom rectangle
    for i in range(len(bottom_vertices)):
        ls.moveTo(*bottom_vertices[i])
        ls.drawTo(*bottom_vertices[(i + 1) % len(bottom_vertices)])
    # Draw top rectangle
    for i in range(len(top_vertices)):
        ls.moveTo(*top_vertices[i])
        ls.drawTo(*top_vertices[(i + 1) % len(top_vertices)])
    # Draw sides
    for i in range(len(bottom_vertices)):
        ls.moveTo(*bottom_vertices[i])
        ls.drawTo(*top_vertices[i])
    # Create and return a node with the segments
    lsnp = NodePath(ls.create())
    lsnp.setTransparency(TransparencyAttrib.MDual)
    lsnp.setLightOff()
    ls_sgm = StaticGeometricModel(lsnp, rgb=None)
    return ls_sgm


def gen_surface(surface_callback, rng, granularity=.01):
    surface_trm = trm_factory.gen_surface(surface_callback, rng, granularity)
    surface_gm = GeometricModel(surface_trm, toggle_twosided=True)
    return surface_gm


if __name__ == "__main__":
    import os
    import math
    import numpy as np
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnygm = GeometricModel(objpath)
    bunnygm.rgba = np.array([0.7, 0.7, 0.0, 1.0])
    bunnygm.attach_to(base)
    bunnygm.show_local_frame()
    rotmat = rm.rotmat_from_axangle(np.array([1, 0, 0]), math.pi / 2.0)
    bunnygm.rotmat = rotmat
    gen_frame().attach_to(base)

    bunnygm1 = bunnygm.copy()
    bunnygm1.rgba = np.array([0.7, 0, 0.7, 1.0])
    bunnygm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnygm1.pos = np.array([0, .01, 0])
    bunnygm1.rotmat = rotmat

    bunnygm2 = bunnygm1.copy()
    bunnygm2.rgba = np.array([0, 0.7, 0.7, 1.0])
    bunnygm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnygm2.pos = np.array([0, .2, 0])
    bunnygm2.rotmat = rotmat
    bunnygm2.set_scale([2, 1, 3])

    bunnygmpoints = bunnygm.sample_surface()
    bunnygm1points = bunnygm1.sample_surface()
    bunnygm2points = bunnygm2.sample_surface()
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
    gen_frame_box(xyz_lengths=np.array([.1, .2, .3]), pos=pos, rotmat=rotmat).attach_to(base)

    base.run()
