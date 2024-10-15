import copy
import uuid
from panda3d.core import NodePath
from direct.showbase.ShowBase import ShowBase
import wrs.modeling.model_collection as mmc
import wrs.modeling._panda_cdhelper as mph
import wrs.modeling._ode_cdhelper as moh
import wrs.basis.robot_math as rm
import wrs.basis.data_adapter as da
import wrs.modeling.geometric_model as mgm
import wrs.modeling.constant as const


# the following two helpers cannot correcty find collision positions, 20211216
# TODO check if it is caused by the bad bullet transformation in moh.update_pose
# import modeling._gimpact_cdhelper as mgh
# import modeling._bullet_cdhelper as mbh


# ============================
# definition of CollisionModel
# ============================

class CollisionModel(mgm.GeometricModel):
    """
    Load an object as a collision model
    Both collison primitives will be generated automatically
    Note: This class heaviliy depends on Panda3D
    Note: Scaling is no longer supported due to complication 20230815
    author: weiwei
    date: 20190312, 20230815
    """

    def __init__(self,
                 initor=None,
                 name="collision_model",
                 cdprim_type=const.CDPrimType.AABB,
                 cdmesh_type=const.CDMeshType.DEFAULT,
                 ex_radius=None,
                 userdef_cdprim_fn=None,
                 toggle_transparency=True,
                 toggle_twosided=False,
                 rgb=rm.const.steel_gray,
                 alpha=1):
        """
        :param initor:
        :param toggle_transparency:
        :param cdprim_type: cdprim type, model_constant.CDPType
        :param cdmesh_type: cdmesh_type, model_constant.CDMType
        :param ex_radius:
        :param name:
        :param userdef_cdprim_fn: the collision primitive will be defined in the provided function
                                           if cdp_type = external;
                                           protocal for the callback function: return NodePath (CollisionNode),
                                           may have multiple CollisionSolid
        date: 20190312, 20201212, 20230814, 20231124
        """

        if isinstance(initor, CollisionModel):
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
            # self.pdndp_core.setMaterialOff()
            # self.pdndp_core.setShaderAuto()
            # collision model
            self.uuid = uuid.uuid4()
            self._ex_radius = initor._ex_radius
            # cd primitive
            self._cdprim_type = initor.cdprim_type
            self._cdprim = copy.deepcopy(initor.cdprim)
            # cd mesh
            self._cdmesh_type = initor.cdmesh_type
            self._cdmesh = copy.deepcopy(initor.cdprim)
            # delays
            self._is_cdprim_delayed = True
            self._is_cdmesh_delayed = True
            # cache for show
            self._cache_for_show = {}
            # others
            self._local_frame = None
        else:
            super().__init__(initor=initor,
                             name=name,
                             toggle_transparency=toggle_transparency,
                             toggle_twosided=toggle_twosided,
                             rgb=rgb,
                             alpha=alpha)
            self.uuid = uuid.uuid4()
            self._ex_radius = ex_radius
            # cd primitive
            self._cdprim_type = cdprim_type
            self._cdprim = self._acquire_cdprim(cdprim_type, ex_radius, userdef_cdprim_fn)
            # cd mesh
            self._cdmesh_type = cdmesh_type
            self._cdmesh = self._acquire_cdmesh(cdmesh_type)
            # delays
            self._is_cdprim_delayed = True
            self._is_cdmesh_delayed = True
            # cache for show
            self._cache_for_show = {}
            # others
            self._local_frame = None

    # delays for cdprim (Panda3D CollisionNode)
    @staticmethod
    def delay_cdprim_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_cdprim_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_cdprim_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_cdprim_delayed:
                self._cdprim.setPosQuat(da.npvec3_to_pdvec3(self.pos), da.npmat3_to_pdquat(self.rotmat))
                self._is_cdprim_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    # delays for cdmesh (OdeTriMeshGeom)
    @staticmethod
    def delay_cdmesh_decorator(method):
        def wrapper(self, *args, **kwargs):
            self._is_cdmesh_delayed = True
            return method(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_cdmesh_decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._is_cdmesh_delayed:
                self._cdmesh.setPosition(da.npvec3_to_pdvec3(self.pos))
                self._cdmesh.setQuaternion(da.npmat3_to_pdquat(self.rotmat))
                self._is_cdmesh_delayed = False
            return method(self, *args, **kwargs)

        return wrapper

    def _acquire_cdmesh(self, cdmesh_type=None, toggle_trm=False):
        """
        step 1: extract vvnf following the specified cdmesh_type
        step 2: pack the vvnf to cdmesh
        :param cdmesh_type:
        :param toggle_trm: return the cdmesh's Trimesh format or not
        :return:
        author: weiwei
        date: 20211215, 20230814
        """
        if self._trm_mesh is None:
            return None
        if cdmesh_type is None:
            cdmesh_type = self.cdmesh_type
        if cdmesh_type == const.CDMeshType.AABB:
            trm_mesh = self._trm_mesh.aabb_bound
        elif cdmesh_type == const.CDMeshType.OBB:
            trm_mesh = self._trm_mesh.obb_bound
        elif cdmesh_type == const.CDMeshType.CONVEX_HULL:
            trm_mesh = self._trm_mesh.convex_hull
        elif cdmesh_type == const.CDMeshType.CYLINDER:
            trm_mesh = self._trm_mesh.cyl_bound
        elif cdmesh_type == const.CDMeshType.DEFAULT:
            trm_mesh = self._trm_mesh
        else:
            raise ValueError("Wrong mesh collision model end_type name!")
        cdmesh = moh.gen_cdmesh(trm_mesh)
        if toggle_trm:
            return cdmesh, trm_mesh
        else:
            return cdmesh

    def _acquire_cdprim(self, cdprim_type=None, thickness=None, userdef_cdprim_fn=None):
        if self._trm_mesh is None:
            return None
        if cdprim_type is None:
            cdprim_type = self.cdprim_type
        if thickness is None:
            thickness = 0.002
        if cdprim_type == const.CDPrimType.AABB:
            pdcndp = mph.gen_aabb_box_pdcndp(self._trm_mesh, ex_radius=thickness)
        elif cdprim_type == const.CDPrimType.OBB:
            pdcndp = mph.gen_obb_box_pdcndp(self._trm_mesh, ex_radius=thickness)
        elif cdprim_type == const.CDPrimType.CAPSULE:
            pdcndp = mph.gen_capsule_pdcndp(self._trm_mesh, ex_radius=thickness)
        elif cdprim_type == const.CDPrimType.CYLINDER:
            pdcndp = mph.gen_cylinder_pdcndp(self._trm_mesh, ex_radius=thickness)
        elif cdprim_type == const.CDPrimType.SURFACE_BALLS:
            pdcndp = mph.gen_surfaceballs_pdcnd(self._trm_mesh, radius=thickness)
        elif cdprim_type == const.CDPrimType.POINT_CLOUD:
            pdcndp = mph.gen_pointcloud_pdcndp(self._trm_mesh, radius=thickness)
        elif cdprim_type == const.CDPrimType.USER_DEFINED:
            if userdef_cdprim_fn is None:
                raise ValueError("User defined functions must provided for user_defined cdprim!")
            pdcndp = userdef_cdprim_fn(ex_radius=thickness)
        else:
            print(cdprim_type)
            raise ValueError("Wrong primitive collision model cdprim_type name!")
        mph.change_cdmask(pdcndp, mph.BITMASK_EXT, action="new", type="both")
        return pdcndp

    @mgm.GeometricModel.pos.setter
    @mgm.GeometricModel.delay_pdndp_pose_decorator
    @delay_cdprim_decorator
    @delay_cdmesh_decorator
    def pos(self, pos: rm.np.ndarray):
        self._pos = pos

    @mgm.GeometricModel.rotmat.setter
    @mgm.GeometricModel.delay_pdndp_pose_decorator
    @delay_cdprim_decorator
    @delay_cdmesh_decorator
    def rotmat(self, rotmat: rm.np.ndarray):
        self._rotmat = rotmat

    @mgm.GeometricModel.homomat.setter
    @mgm.GeometricModel.delay_pdndp_pose_decorator
    @delay_cdprim_decorator
    @delay_cdmesh_decorator
    def homomat(self, homomat: rm.np.ndarray):
        self._pos = homomat[:3, 3]
        self._rotmat = homomat[:3, :3]

    @mgm.GeometricModel.pose.setter
    @mgm.GeometricModel.delay_pdndp_pose_decorator
    @delay_cdprim_decorator
    @delay_cdmesh_decorator
    def pose(self, pose):
        """
        :param pose: tuple or list containing an npvec3 and an npmat3
        :return:
        """
        self._pos = pose[0]
        self._rotmat = pose[1]

    @property
    def cdmesh_type(self):
        return self._cdmesh_type

    @property
    @update_cdmesh_decorator
    def cdmesh(self):
        return self._cdmesh

    @property
    def cdprim_type(self):
        return self._cdprim_type

    @property
    @update_cdprim_decorator
    def cdprim(self):
        return self._cdprim

    @delay_cdmesh_decorator
    def change_cdmesh_type(self, cdmesh_type):
        self._cdmesh = self._acquire_cdmesh(cdmesh_type)
        self._cdmesh_type = cdmesh_type
        # update if show_cdmesh is toggled on
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()
            _, cdmesh_trm_model = self._acquire_cdmesh(toggle_trm=True)
            self._cache_for_show["cdmesh"] = mph.gen_pdndp_wireframe(trm_model=cdmesh_trm_model)
            self._cache_for_show["cdmesh"].reparentTo(self.pdndp)
            mph.toggle_show_collision_node(self._cache_for_show["cdmesh"], toggle_show_on=True)

    @delay_cdprim_decorator
    def change_cdprim_type(self, cdprim_type, ex_radius=None, userdef_cdprim_fn=None):
        if ex_radius is not None:
            self._ex_radius = ex_radius
        self._cdprim = self._acquire_cdprim(cdprim_type=cdprim_type,
                                            thickness=ex_radius,
                                            userdef_cdprim_fn=userdef_cdprim_fn)
        self._cdprim_type = cdprim_type
        # update if show_primitive is toggled on
        if "cdprim" in self._cache_for_show:
            self._cache_for_show["cdprim"].removeNode()
            self._cache_for_show["cdprim"] = self.copy_reference_cdprim()
            self._cache_for_show["cdprim"].reparentTo(self.pdndp)
            mph.toggle_show_collision_node(self._cache_for_show["cdprim"], toggle_show_on=True)

    @update_cdprim_decorator
    def attach_cdprim_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._cdprim.reparentTo(target.render)
        elif isinstance(target, mgm.StaticGeometricModel):  # prepared for decorations like local frames
            self._cdprim.reparentTo(target.pdndp)
        elif isinstance(target, NodePath):
            # print(self._cdprim.getPos(), self._cdprim.getMat())
            self._cdprim.reparentTo(target)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, NodePath!")
        return self._cdprim

    def detach_cdprim(self):
        self._cdprim.detachNode()

    def copy_reference_cdmesh(self):
        """
        return a copy of the cdmesh without updating to the current mcm pose
        "reference" means the returned cdmesh copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20211215, 20230815
        """
        return_cdmesh = copy.deepcopy(self._cdmesh)
        # clear rotmat
        return_cdmesh.setPosition(da.npvec3_to_pdvec3(rm.np.zeros(3)))
        return_cdmesh.setRotation(da.npmat3_to_pdmat3(rm.np.eye(3)))
        return return_cdmesh

    @update_cdmesh_decorator
    def copy_transformed_cdmesh(self):
        """
        return a copy of the cdmesh without updating to the current mcm pose
        "reference" means the returned cdmesh copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20211215, 20230815
        """
        return_cdmesh = copy.deepcopy(self._cdmesh)
        return return_cdmesh

    def copy_reference_cdprim(self):
        """
        return a copy of the cdprim without updating to the current mcm pose
        "reference" means the returned cdprim copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20230815
        """
        return_cdprim = copy.deepcopy(self._cdprim)
        return_cdprim.clearMat()
        return return_cdprim

    @update_cdprim_decorator
    def copy_transformed_cdprim(self):
        """
        return a copy of the cdprim without updating to the current mcm pose
        "reference" means the returned cdprim copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20230815
        """
        return_cdprim = copy.deepcopy(self._cdprim)
        return return_cdprim

    def is_pcdwith(self, cmodel, toggle_contacts=False):
        """
        Is the primitives of this mcm collide with the primitives of the given mcm
        :param cmodel: one or a list of Collision Model object
        :param toggle_contacts: return a list of contact points if toggle_contacts is True
        author: weiwei
        date: 20201116
        """
        return mph.is_collided(self, cmodel, toggle_contacts=toggle_contacts)

    @mgm.GeometricModel.update_pdndp_pose_decorator
    def attach_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._pdndp.reparentTo(target.render)
        elif isinstance(target, mgm.StaticGeometricModel):  # prepared for decorations like local frames
            self._pdndp.reparentTo(target.pdndp)
        elif isinstance(target, mmc.ModelCollection):
            target.add_cm(self)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, ModelCollection!")

    def show_cdprim(self):
        if "cdprim" in self._cache_for_show:
            self._cache_for_show["cdprim"].removeNode()
        self._cache_for_show["cdprim"] = self.copy_reference_cdprim()
        self._cache_for_show["cdprim"].reparentTo(self.pdndp)
        mph.toggle_show_collision_node(self._cache_for_show["cdprim"], toggle_show_on=True)

    def unshow_cdprim(self):
        if "cdprim" in self._cache_for_show:
            self._cache_for_show["cdprim"].removeNode()

    def show_cdmesh(self):
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()
        _, cdmesh_trm_model = self._acquire_cdmesh(toggle_trm=True)
        self._cache_for_show["cdmesh"] = mph.gen_pdndp_wireframe(trm_model=cdmesh_trm_model)
        self._cache_for_show["cdmesh"].reparentTo(self.pdndp)
        mph.toggle_show_collision_node(self._cache_for_show["cdmesh"], toggle_show_on=True)

    def unshow_cdmesh(self):
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()

    def is_mcdwith(self, cmodel_list, toggle_contacts=False):
        """
        Is the mesh of the mcm collide with the mesh of the given mcm
        :param cmodel_list: one or a list of Collision Model object
        :param toggle_contacts: return a list of contact points if toggle_contacts is True
        author: weiwei
        date: 20201116
        """
        return moh.is_collided(self, cmodel_list, toggle_contacts=toggle_contacts)

    def ray_hit(self, spos, epos, option="all"):
        """
        check the intersection between segment point_from-point_to and the mesh
        :param spos: 1x3 nparray
        :param epos:
        :param option: "all" or “closest"
        :return:
        author: weiwei
        date: 20210504
        """
        if option == "all":
            return moh.rayhit_all(spos, epos, self)
        elif option == "closest":
            return moh.rayhit_closet(spos, epos, self)

    def copy(self):
        cmodel = CollisionModel(self)
        cmodel.pos = self.pos
        cmodel.rotmat = self.rotmat
        cmodel.change_cdmesh_type(cdmesh_type=self.cdmesh_type)
        cmodel.change_cdprim_type(cdprim_type=self.cdprim_type, ex_radius=self._ex_radius)  # TODO user_defined_fn
        return cmodel


# ======================================================
# helper functions for creating various collision models
# ======================================================


def gen_surface_barrier(pos_z=.0, rgb=rm.const.tab20_list[14], alpha=1):
    return gen_box(rm.np.array([5, 5, 1]), rm.np.array([.0, .0, -.5 + pos_z]), rgb=rgb, alpha=alpha)


def gen_box(xyz_lengths=rm.np.array([.1, .1, .1]),
            pos=rm.np.zeros(3),
            rotmat=rm.np.eye(3),
            rgb=rm.const.tab20_list[14],
            alpha=1):
    """
    :param xyz_lengths:
    :param pos:
    :param rotmat:
    :param rgb:
    :param alpha:
    :return:
    author: weiwei
    date: 20201202, 20240303
    """
    box_sgm = mgm.gen_box(xyz_lengths=xyz_lengths, rgb=rgb, alpha=alpha)
    box_cm = CollisionModel(box_sgm)
    box_cm.pose = [pos, rotmat]
    return box_cm


def gen_sphere(pos=rm.np.array([0, 0, 0]), radius=0.01, rgb=rm.const.tab20_list[10], alpha=1):
    """
    :param pos:
    :param radius:
    :param rgb:
    :param alpha:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_sgm = mgm.gen_sphere(pos=pos, radius=radius, rgb=rgb, alpha=alpha)
    sphere_cm = CollisionModel(sphere_sgm)
    return sphere_cm


def gen_stick(spos=rm.np.array([.0, .0, .0]),
              epos=rm.np.array([.0, .0, .1]),
              radius=.0025,
              type="rect",
              rgb=rm.const.tab20_list[10],
              alpha=1,
              n_sec=16):
    """
    :param spos:
    :param epos:
    :param radius:
    :param type: "rect"
    :param rgb:
    :param alpha:
    :param n_sec:
    :return: 20210328
    """
    center_rotmat = rm.rotmat_between_vectors(v1=rm.np.array([0, 0, 1]), v2=epos-spos)
    length = rm.np.linalg.norm(epos - spos)
    stick_sgm = mgm.gen_stick(spos=rm.np.array([.0, .0, .0]), epos=rm.np.array([.0, .0, 1.0]) * length,
                              radius=radius, type=type, rgb=rgb, alpha=alpha, n_sec=n_sec)
    stick_cm = CollisionModel(stick_sgm, cdprim_type=const.CDPrimType.CYLINDER)
    stick_cm.pose = [spos, center_rotmat]
    return stick_cm


if __name__ == "__main__":
    import os
    import math
    import time
    import numpy as np
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.3, .3, .3], lookat_pos=[0, 0, 0], toggle_debug=True)
    mgm.gen_frame().attach_to(base)
    box = gen_stick(spos=np.array([0,0,-1]), epos=np.array([1,0,0]), radius=.1)
    box.attach_to(base)
    box.show_cdprim()
    objpath = os.path.join(os.path.dirname(rm.__file__), "objects", "bunnysim.stl")
    bunnycm = CollisionModel(objpath, cdprim_type=const.CDPrimType.CAPSULE)
    bunnycm.rgba = rm.np.array([0.7, 0.7, 0.0, .2])
    bunnycm.show_local_frame()
    bunnycm.attach_to(base)
    bunnycm.change_cdmesh_type(cdmesh_type=const.CDMeshType.CYLINDER)
    bunnycm.show_cdprim()
    bunnycm1 = CollisionModel(objpath, cdprim_type=const.CDPrimType.CYLINDER)
    bunnycm1.rgba = rm.np.array([0.7, 0, 0.7, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.pos = rm.np.array([0, .01, 0])
    bunnycm1.rotmat = rotmat
    bunnycm1.attach_to(base)
    bunnycm1.show_cdprim()
    tic = time.time()
    result, contacts = bunnycm.is_pcdwith(bunnycm1, toggle_contacts=True)
    toc = time.time()
    print("mesh cd cost: ", toc - tic)
    print(result)
    ct_gmodel = mgm.GeometricModel(contacts)
    ct_gmodel.attach_to(base)
    base.run()