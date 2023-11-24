import copy
import math
import numpy as np
from visualization.panda.world import ShowBase
from panda3d.core import NodePath, CollisionNode, CollisionTraverser, CollisionHandlerQueue, BitMask32
import basis.data_adapter as da
import basis.constant as bc
import modeling.geometric_model as mgm
import modeling.model_collection as mmc
import modeling._panda_cdhelper as mph
import modeling._ode_cdhelper as moh
# import modeling._bullet_cdhelper as mbh
import modeling.constant as mc
import uuid


# the following two helpers cannot correcty find collision positions, 20211216
# TODO check if it is caused by the bad bullet transformation in moh.update_pose
# import modeling._gimpact_cdhelper as mgh
# import modeling._bullet_cdhelper as mbh

# ==============================================
# delays for cdprimitive (Panda3D CollisionNode)
# ==============================================

def delay_cdprimitive_decorator(method):
    def wrapper(self, *args, **kwargs):
        self._is_cdp_delayed = True
        return method(self, *args, **kwargs)

    return wrapper


def update_cdprimitive_decorator(method):
    def wrapper(self, *args, **kwargs):
        if self._is_cdp_delayed:
            print("cdprimitive_delayed, update: ", self.pos, self.rotmat)
            self._cdp.setPosQuat(da.npvec3_to_pdvec3(self.pos), da.npmat3_to_pdquat(self.rotmat))
            self._is_cdp_delayed = False
        return method(self, *args, **kwargs)

    return wrapper


# ==================================
# delays for cdmesh (OdeTriMeshGeom)
# ==================================

def delay_cdmesh_decorator(method):
    def wrapper(self, *args, **kwargs):
        self._is_cdm_delayed = True
        return method(self, *args, **kwargs)

    return wrapper


def update_cdmesh_decorator(method):
    def wrapper(self, *args, **kwargs):
        if self._is_cdp_delayed:
            self._cdm.setPosition(da.npvec3_to_pdvec3(self.pos))
            self._cdm.setQuaternion(da.npmat3_to_pdquat(self.rotmat))
            self._is_cdp_delayed = False
        return method(self, *args, **kwargs)

    return wrapper


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
                 initor,
                 cdp_type=mc.CDPType.BOX,
                 cdm_type=mc.CDMType.DEFAULT,
                 expand_radius=None,
                 name="collision_model",
                 userdef_cdp_fn=None,
                 toggle_transparency=True,
                 toggle_twosided=False):
        """
        :param initor:
        :param toggle_transparency:
        :param cdp_type: cdprimitive type, model_constant.CDPType
        :param cdm_type: cdmesh_type, model_constant.CDMType
        :param expand_radius:
        :param name:
        :param userdef_cdp_fn: the collision primitive will be defined in the provided function
                                           if cdp_type = external;
                                           protocal for the callback function: return NodePath (CollisionNode),
                                           may have multiple CollisionSolid
        date: 20190312, 20201212, 20230814, 20231124
        """
        if isinstance(initor, CollisionModel):
            self._name = copy.deepcopy(initor.name)
            self._file_path = copy.deepcopy(initor.file_path)
            self._trm_mesh = copy.deepcopy(initor.trm_mesh)
            self._pdndp = copy.deepcopy(initor.pdndp)
            self._cdm_type = copy.deepcopy(initor.cdmesh_type)
            self._cdm = copy.deepcopy(initor.cdmesh)
            self._cdp_type = copy.deepcopy(initor.cdprimitive_type)
            self._cdp = copy.deepcopy(initor.cdprimitive)
            self._cache_for_show = copy.deepcopy(initor._cache_for_show)
            self._local_frame = copy.deepcopy(initor.local_frame)
            self._is_geometry_delayed = copy.deepcopy(initor._is_geometry_delayed)
            self._is_cdp_delayed = copy.deepcopy(initor._is_cdp_delayed)
            self._is_cdm_delayed = copy.deepcopy(initor._is_cdm_delayed)
        else:
            super().__init__(initor=initor,
                             name=name,
                             toggle_transparency=toggle_transparency,
                             toggle_twosided=toggle_twosided)
            self._exp_radius = expand_radius
            # cd primitive
            self._cdp_type = cdp_type
            self._cdp = self._acquire_cdp(cdp_type,
                                          expand_radius,
                                          userdef_cdp_fn)
            # cd mesh
            self._cdm_type = cdm_type
            self._cdm = self._acquire_cdm(cdm_type)
            # delays
            self._is_cdp_delayed = False
            self._is_cdm_delayed = False
            # cache for show
            self._cache_for_show = {}
            # others
            self._local_frame = None

    def _acquire_cdm(self, cdmesh_type=None, toggle_trm=False):
        """
        step 1: extract vvnf following the specified cdm_type
        step 2: pack the vvnf to cdmesh
        :param cdmesh_type:
        :param toggle_trm: return the cdmesh's Trimesh format or not
        :return:
        author: weiwei
        date: 20211215, 20230814
        """
        if cdmesh_type is None:
            cdmesh_type = self.cdmesh_type
        if cdmesh_type == mc.CDMType.AABB:
            trm_mesh = self.trm_mesh.aabb_bound
        elif cdmesh_type == mc.CDMType.OBB:
            trm_mesh = self.trm_mesh.obb_bound
        elif cdmesh_type == mc.CDMType.CONVEX_HULL:
            trm_mesh = self.trm_mesh.convex_hull
        elif cdmesh_type == mc.CDMType.CYLINDER:
            trm_mesh = self.trm_mesh.cyl_bound
        elif cdmesh_type == mc.CDMType.DEFAULT:
            trm_mesh = self.trm_mesh
        else:
            raise ValueError("Wrong mesh collision model end_type name!")
        cdmesh = moh.gen_cdmesh(trm_mesh)
        if toggle_trm:
            return cdmesh, trm_mesh
        else:
            return cdmesh

    def _acquire_cdp(self, cdprimitive_type=None, thickness=None, userdefined_cdprimitive_fn=None):
        if cdprimitive_type is None:
            cdprimitive_type = self.cdprimitive_type
        if thickness is None:
            thickness = 0.002
        if cdprimitive_type == mc.CDPType.BOX:
            pdcndp = mph.gen_box_pdcndp(self.trm_mesh, ex_radius=thickness)
        elif cdprimitive_type == mc.CDPType.CAPSULE:
            pdcndp = mph.gen_capsule_pdcndp(self.trm_mesh, ex_radius=thickness)
        elif cdprimitive_type == mc.CDPType.CYLINDER:
            pdcndp = mph.gen_cylinder_pdcndp(self.trm_mesh, ex_radius=thickness)
        elif cdprimitive_type == mc.CDPType.SURFACE_BALLS:
            pdcndp = mph.gen_surfaceballs_pdcnd(self.trm_mesh, radius=thickness)
        elif cdprimitive_type == mc.CDPType.POINT_CLOUD:
            pdcndp = mph.gen_pointcloud_pdcndp(self.trm_mesh, radius=thickness)
        elif cdprimitive_type == mc.CDPType.USER_DEFINED:
            if userdefined_cdprimitive_fn is None:
                raise ValueError("User defined functions must provided for user_defined cdprimitive!")
            pdcndp = userdefined_cdprimitive_fn(ex_radius=thickness)
        else:
            raise ValueError("Wrong primitive collision model end_type name!")
        mph.change_cdmask(pdcndp, mph.BITMASK_EXT, action="new", type="both")
        return pdcndp

    @property
    def cdmesh_type(self):
        return self._cdm_type

    @property
    @update_cdmesh_decorator
    def cdmesh(self):
        return self._cdm

    @property
    def cdprimitive_type(self):
        return self._cdp_type

    @property
    @update_cdprimitive_decorator
    def cdprimitive(self):
        return self._cdp

    @delay_cdmesh_decorator
    def change_cdmesh_type(self, cdmesh_type):
        self._cdm = self._acquire_cdm(cdmesh_type)
        self._cdm_type = cdmesh_type
        # update if show_cdmesh is toggled on
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()
            _, cdmesh_trm_model = self._acquire_cdm(toggle_trm=True)
            self._cache_for_show["cdmesh"] = mph.gen_pdndp_wireframe(trm_model=cdmesh_trm_model)
            self._cache_for_show["cdmesh"].reparentTo(self.pdndp)
            mph.toggle_show_collision_node(self._cache_for_show["cdmesh"], toggle_show_on=True)

    @delay_cdprimitive_decorator
    def change_cdprimitive_type(self, cdprimitive_type, expand_radius=None, userdefined_cdprimitive_fn=None):
        if expand_radius is not None:
            self._exp_radius = expand_radius
        self._cdp = self._acquire_cdp(cdprimitive_type=cdprimitive_type,
                                      thickness=expand_radius,
                                      userdefined_cdprimitive_fn=userdefined_cdprimitive_fn)
        self._cdp_type = cdprimitive_type
        # update if show_primitive is toggled on
        if "cdprimitive" in self._cache_for_show:
            self._cache_for_show["cdprimitive"].removeNode()
            self._cache_for_show["cdprimitive"] = self.copy_reference_cdprimitive()
            self._cache_for_show["cdprimitive"].reparentTo(self.pdndp)
            mph.toggle_show_collision_node(self._cache_for_show["cdprimitive"], toggle_show_on=True)

    @update_cdprimitive_decorator
    def attach_cdprimitive_to(self, target):
        if isinstance(target, ShowBase):
            # for rendering to base.render
            self._cdp.reparentTo(target.render)
        elif isinstance(target, mgm.StaticGeometricModel):  # prepared for decorations like local frames
            self._cdp.reparentTo(target.pdndp)
        elif isinstance(target, NodePath):
            self._cdp.reparentTo(target)
        else:
            raise ValueError("Acceptable: ShowBase, StaticGeometricModel, NodePath!")
        return self._cdp

    def copy_reference_cdmesh(self):
        """
        return a copy of the cdmesh without updating to the current mcm pose
        "reference" means the returned cdmesh copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20211215, 20230815
        """
        return_cdmesh = copy.deepcopy(self._cdm)
        # clear rotmat
        return_cdmesh.setPosition(da.npvec3_to_pdvec3(np.zeros(3)))
        return_cdmesh.setRotation(da.npmat3_to_pdmat3(np.eye(3)))
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
        return_cdmesh = copy.deepcopy(self._cdm)
        return return_cdmesh

    def copy_reference_cdprimitive(self):
        """
        return a copy of the cdprimitive without updating to the current mcm pose
        "reference" means the returned cdprimitive copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20230815
        """
        return_cdp = copy.deepcopy(self._cdp)
        # print(return_cdprimitive.getPos(), return_cdprimitive.getMat())
        # return_cdprimitive.setPosQuat(da.npvec3_to_pdvec3(np.zeros(3)), da.npmat3_to_pdquat(np.eye(3)))
        return_cdp.clearMat()
        return return_cdp

    @update_cdprimitive_decorator
    def copy_transformed_cdprimitive(self):
        """
        return a copy of the cdprimitive without updating to the current mcm pose
        "reference" means the returned cdprimitive copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20230815
        """
        return_cdprimitive = copy.deepcopy(self._cdp)
        return return_cdprimitive

    def is_pcdwith(self, cmodel, toggle_contacts=False):
        """
        Is the primitives of this mcm collide with the primitives of the given mcm
        :param cmodel: one or a list of Collision Model object
        :param toggle_contacts: return a list of contact points if toggle_contacts is True
        author: weiwei
        date: 20201116
        """
        return mph.is_collided(self, cmodel, toggle_contacts=toggle_contacts)

    def show_cdprimitive(self):
        if "cdprimitive" in self._cache_for_show:
            self._cache_for_show["cdprimitive"].removeNode()
        self._cache_for_show["cdprimitive"] = self.copy_reference_cdprimitive()
        self._cache_for_show["cdprimitive"].reparentTo(self.pdndp)
        mph.toggle_show_collision_node(self._cache_for_show["cdprimitive"], toggle_show_on=True)

    def unshow_cdprimitive(self):
        if "cdprimitive" in self._cache_for_show:
            self._cache_for_show["cdprimitive"].removeNode()

    def show_cdmesh(self):
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()
        _, cdmesh_trm_model = self._acquire_cdm(toggle_trm=True)
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
            contact_points, contact_normals = moh.rayhit_all(spos, epos, self)
            return contact_points, contact_normals
        elif option == "closest":
            contact_point, contact_normal = moh.rayhit_closet(spos, epos, self)
            return contact_point, contact_normal

    def copy(self):
        return CollisionModel(self)

    def __deepcopy__(self):
        """
        this function helps make sure the uuid is unique
        :return:
        """
        return CollisionModel(self)


# ======================================================
# helper functions for creating various collision models
# ======================================================


def gen_box(extent=np.array([.1, .1, .1]), homomat=np.eye(4), rgba=np.array([1, 0, 0, 1])):
    """
    :param extent:
    :param homomat:
    :return:
    author: weiwei
    date: 20201202
    """
    box_sgm = mgm.gen_box(extent=extent, homomat=homomat, rgba=rgba)
    box_cm = CollisionModel(box_sgm)
    return box_cm


def gen_sphere(pos=np.array([0, 0, 0]), radius=0.01, rgba=[1, 0, 0, 1]):
    """
    :param pos:
    :param radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20161212tsukuba, 20191228osaka
    """
    sphere_sgm = mgm.gen_sphere(pos=pos, radius=radius, rgba=rgba)
    sphere_cm = CollisionModel(sphere_sgm)
    return sphere_cm


def gen_stick(spos=np.array([.0, .0, .0]),
              epos=np.array([.0, .0, .1]),
              radius=.0025,
              type="round",
              rgba=np.array([1, 0, 0, 1]),
              n_sec=8):
    """
    :param spos:
    :param epos:
    :param radius:
    :param rgba:
    :return:
    author: weiwei
    date: 20210328
    """
    stick_sgm = mgm.gen_stick(spos=spos, epos=epos, radius=radius, type=type, rgba=rgba, n_sec=n_sec)
    stick_cm = CollisionModel(stick_sgm)
    return stick_cm


if __name__ == "__main__":
    import os
    import math
    import time
    import numpy as np
    import basis
    import basis.robot_math as rm
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[.3, .3, .3], lookat_pos=[0, 0, 0], toggle_debug=True)
    mgm.gen_frame().attach_to(base)

    objpath = os.path.join(basis.__path__[0], "objects", "bunnysim.stl")
    bunnycm = CollisionModel(objpath, cdp_type=mc.CDPType.CAPSULE)
    bunnycm.rgba = np.array([0.7, 0.7, 0.0, .2])
    bunnycm.show_local_frame()
    bunnycm.attach_to(base)
    bunnycm.change_cdmesh_type(mc.CDMType.CYLINDER)
    bunnycm.show_cdprimitive()

    bunnycm1 = CollisionModel(objpath, cdp_type=mc.CDPType.CYLINDER)
    bunnycm1.rgba = np.array([0.7, 0, 0.7, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.pos = np.array([0, .01, 0])
    bunnycm1.rotmat = rotmat
    bunnycm1.attach_to(base)
    bunnycm1.show_cdprimitive()

    tic = time.time()
    result, contacts = bunnycm.is_pcdwith(bunnycm1, toggle_contacts=True)
    toc = time.time()
    print("mesh cd cost: ", toc - tic)
    print(result)
    ct_gmodel = mgm.GeometricModel(contacts)
    ct_gmodel.attach_to(base)
    base.run()
