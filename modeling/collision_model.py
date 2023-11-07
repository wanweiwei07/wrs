import copy
import math
import numpy as np
from panda3d.core import BitMask32
from visualization.panda.world import ShowBase
import basis.data_adapter as da
import modeling.geometric_model as mgm
import modeling.model_collection as mmc
import modeling._panda_cdhelper as pcd_helper
import modeling._ode_cdhelper as mcd_helper
# import modeling._bullet_cdhelper as mcd_helper
import modeling.constant as mc


# the following two helpers cannot correcty find collision positions, 20211216
# TODO check if it is caused by the bad bullet transformation in mcd_helper.update_pose
# import modeling._gimpact_cdhelper as mcd_helper
# import modeling._bullet_cdhelper as mcd_helper


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
                 cdprimitive_type=mc.CDPrimitiveType.BOX,
                 cdmesh_type=mc.CDMeshType.DEFAULT,
                 expand_radius=None,
                 name="cm",
                 userdefined_cdprimitive_fn=None,
                 toggle_transparency=True,
                 toggle_twosided=False):
        """
        :param initor:
        :param toggle_transparency:
        :param cdprimitive_type: box, ball, capsule, point_cloud, user_defined
        :param cdmesh_type: aabb, obb, convex_hull, default(triangulation)
        :param expand_radius:
        :param name:
        :param userdefined_cdprimitive_fn: the collision primitive will be defined in the provided function
                                           if cdprimitive_type = external;
                                           protocal for the callback function: return CollisionNode,
                                           may have multiple CollisionSolid
        date: 20190312, 20201212, 20230814
        """
        if isinstance(initor, CollisionModel):
            self._name = copy.deepcopy(initor.name)
            self._file_path = copy.deepcopy(initor.file_path)
            self._trm_mesh = copy.deepcopy(initor.trm_mesh)
            self._pdndp = copy.deepcopy(initor.pdndp)
            self._cdmesh_type = copy.deepcopy(initor.cdmesh_type)
            self._cdmesh = initor.copy_reference_cdmesh()
            self._cdprimitive_type = copy.deepcopy(initor.cdprimitive_type)
            self._cdprimitive = initor.copy_reference_cdprimitive()
            self._cache_for_show = copy.deepcopy(initor._cache_for_show)
            self._local_frame = copy.deepcopy(initor.local_frame)
        else:
            super().__init__(initor=initor,
                             name=name,
                             toggle_transparency=toggle_transparency,
                             toggle_twosided=toggle_twosided)
            # cd primitive
            self._cdprimitive = self._acquire_cdprimitive(cdprimitive_type,
                                                          expand_radius,
                                                          userdefined_cdprimitive_fn)
            self._cdprimitive_type = cdprimitive_type
            # cd mesh
            self._cdmesh = self._acquire_cdmesh(cdmesh_type)
            self._cdmesh_type = cdmesh_type
            # cache for show
            self._cache_for_show = {}
            # others
            self._local_frame = None

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
        if cdmesh_type is None:
            cdmesh_type = self.cdmesh_type
        if cdmesh_type == mc.CDMeshType.AABB:
            trm_mesh = self.trm_mesh.aabb_bound
        elif cdmesh_type == mc.CDMeshType.OBB:
            trm_mesh = self.trm_mesh.obb_bound
        elif cdmesh_type == mc.CDMeshType.CONVEX_HULL:
            trm_mesh = self.trm_mesh.convex_hull
        elif cdmesh_type == mc.CDMeshType.CYLINDER:
            trm_mesh = self.trm_mesh.cyl_bound
        elif cdmesh_type == mc.CDMeshType.DEFAULT:
            trm_mesh = self.trm_mesh
        else:
            raise ValueError("Wrong mesh collision model end_type name!")
        cdmesh = mcd_helper.gen_cdmesh(trm_mesh)
        if toggle_trm:
            return cdmesh, trm_mesh
        else:
            return cdmesh

    def _acquire_cdprimitive(self, cdprimitive_type=None, thickness=None, userdefined_cdprimitive_fn=None):
        if cdprimitive_type is None:
            cdprimitive_type = self.cdprimitive_type
        if thickness is None:
            thickness = 0.002
        if cdprimitive_type == mc.CDPrimitiveType.BOX:
            pdcndp = pcd_helper.gen_box_pdcndp(self.trm_mesh, ex_radius=thickness)
        elif cdprimitive_type == mc.CDPrimitiveType.CAPSULE:
            pdcndp = pcd_helper.gen_capsule_pdcndp(self.trm_mesh, ex_radius=thickness)
        elif cdprimitive_type == mc.CDPrimitiveType.CYLINDER:
            pdcndp = pcd_helper.gen_cyl_pdcndp(self.trm_mesh, ex_radius=thickness)
        elif cdprimitive_type == mc.CDPrimitiveType.SURFACE_BALLS:
            pdcndp = pcd_helper.gen_surfaceballs_pdcnd(self.trm_mesh, radius=thickness)
        elif cdprimitive_type == mc.CDPrimitiveType.POINT_CLOUD:
            pdcndp = pcd_helper.gen_pointcloud_pdcndp(self.trm_mesh, radius=thickness)
        elif cdprimitive_type == mc.CDPrimitiveType.USER_DEFINED:
            if userdefined_cdprimitive_fn is None:
                raise ValueError("User defined functions must provided for user_defined cdprimitive!")
            pdcndp = userdefined_cdprimitive_fn(ex_radius=thickness)
        else:
            raise ValueError("Wrong primitive collision model end_type name!")
        pcd_helper.update_collide_mask(pdcndp, BitMask32(2 ** 31))
        return pdcndp

    @property
    def cdmesh_type(self):
        return self._cdmesh_type

    @property
    def cdprimitive_type(self):
        return self._cdprimitive_type

    def change_cdmesh_type(self, cdmesh_type):
        self._cdmesh = self._acquire_cdmesh(cdmesh_type)
        self._cdmesh_type = cdmesh_type
        # update if show_cdmesh is toggled on
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()
            _, cdmesh_trm_model = self._acquire_cdmesh(toggle_trm=True)
            self._cache_for_show["cdmesh"] = pcd_helper.gen_pdndp_wireframe(trm_model=cdmesh_trm_model)
            self._cache_for_show["cdmesh"].reparentTo(self.pdndp)
            pcd_helper.toggle_show_collision_node(self._cache_for_show["cdmesh"], toggle_value=True)

    def change_cdprimitive_type(self, cdprimitive_type, thickness=None, userdefined_cdprimitive_fn=None):
        self._cdprimitive = self._acquire_cdprimitive(cdprimitive_type=cdprimitive_type,
                                                      thickness=thickness,
                                                      userdefined_cdprimitive_fn=userdefined_cdprimitive_fn)
        self._cdprimitive_type = cdprimitive_type
        # update if show_primitive is toggled on
        if "cdprimitive" in self._cache_for_show:
            self._cache_for_show["cdprimitive"].removeNode()
            self._cache_for_show["cdprimitive"] = self.copy_reference_cdprimitive()
            self._cache_for_show["cdprimitive"].reparentTo(self.pdndp)
            pcd_helper.toggle_show_collision_node(self._cache_for_show["cdprimitive"], toggle_value=True)

    def copy_reference_cdmesh(self):
        """
        return a copy of the cdmesh without updating to the current cm pose
        "reference" means the returned cdmesh copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20211215, 20230815
        """
        return_cdmesh = copy.deepcopy(self._cdmesh)
        return return_cdmesh

    def copy_transformed_cdmesh(self):
        """
        return a copy of the cdmesh without updating to the current cm pose
        "reference" means the returned cdmesh copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20211215, 20230815
        """
        return_cdmesh = copy.deepcopy(self._cdmesh)
        mcd_helper.update_pose(return_cdmesh, self)
        return return_cdmesh

    def copy_reference_cdprimitive(self):
        """
        return a copy of the cdprimitive without updating to the current cm pose
        "reference" means the returned cdprimitive copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20230815
        """
        return_cdprimitive = copy.deepcopy(self._cdprimitive)
        return return_cdprimitive

    def copy_transformed_cdprimitive(self):
        """
        return a copy of the cdprimitive without updating to the current cm pose
        "reference" means the returned cdprimitive copy is not affected by tranformation
        :return:
        author: weiwei
        date: 20230815
        """
        return_cdprimitive = copy.deepcopy(self._cdprimitive)
        return_cdprimitive.setMat(self.pdndp.getMat())
        return return_cdprimitive

    def set_pos(self, pos: np.ndarray = np.zeros(3)):
        self._pdndp.setPos(pos[0], pos[1], pos[2])

    def set_rotmat(self, rotmat: np.ndarray = np.eye(3)):
        self._pdndp.setQuat(da.npmat3_to_pdquat(rotmat))

    def set_pose(self, pos: np.ndarray = np.zeros(3), rotmat: np.ndarray = np.eye(3)):
        self._pdndp.setPosQuat(da.npvec3_to_pdvec3(pos), da.npmat3_to_pdquat(rotmat))

    def set_homomat(self, npmat4):
        self._pdndp.setPosQuat(da.npvec3_to_pdvec3(npmat4[:3, 3]), da.npmat3_to_pdquat(npmat4[:3, :3]))

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

    def is_pcdwith(self, objcm, toggle_contacts=False):
        """
        Is the primitives of this cm collide with the primitives of the given cm
        :param objcm: one or a list of Collision Model object
        :param toggle_contacts: return a list of contact points if toggle_contacts is True
        author: weiwei
        date: 20201116
        """
        return pcd_helper.is_collided(self, objcm, toggle_contacts=toggle_contacts)

    def attach_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._pdndp.reparentTo(obj.render)
        elif isinstance(obj, mmc.ModelCollection):
            obj.add_cm(self)
        else:
            print("Must be ShowBase, modeling.StaticGeometricModel, GeometricModel, CollisionModel, "
                  "or CollisionModelCollection!")

    def detach(self):
        self._pdndp.detachNode()

    def show_cdprimitive(self):
        if "cdprimitive" in self._cache_for_show:
            self._cache_for_show["cdprimitive"].removeNode()
        self._cache_for_show["cdprimitive"] = self.copy_reference_cdprimitive()
        self._cache_for_show["cdprimitive"].reparentTo(self.pdndp)
        pcd_helper.toggle_show_collision_node(self._cache_for_show["cdprimitive"], toggle_value=True)

    def unshow_cdprimitive(self):
        if "cdprimitive" in self._cache_for_show:
            self._cache_for_show["cdprimitive"].removeNode()

    def show_cdmesh(self):
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()
        _, cdmesh_trm_model = self._acquire_cdmesh(toggle_trm=True)
        self._cache_for_show["cdmesh"] = pcd_helper.gen_pdndp_wireframe(trm_model=cdmesh_trm_model)
        self._cache_for_show["cdmesh"].reparentTo(self.pdndp)
        pcd_helper.toggle_show_collision_node(self._cache_for_show["cdmesh"], toggle_value=True)

    def unshow_cdmesh(self):
        if "cdmesh" in self._cache_for_show:
            self._cache_for_show["cdmesh"].removeNode()

    def is_mcdwith(self, objcm_list, toggle_contacts=False):
        """
        Is the mesh of the cm collide with the mesh of the given cm
        :param objcm_list: one or a list of Collision Model object
        :param toggle_contacts: return a list of contact points if toggle_contacts is True
        author: weiwei
        date: 20201116
        """
        return mcd_helper.is_collided(self, objcm_list, toggle_contacts=toggle_contacts)

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
            contact_points, contact_normals = mcd_helper.rayhit_all(spos, epos, self)
            return contact_points, contact_normals
        elif option == "closest":
            contact_point, contact_normal = mcd_helper.rayhit_closet(spos, epos, self)
            return contact_point, contact_normal

    def copy(self):
        return CollisionModel(self)


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
    bunnycm = CollisionModel(objpath, cdprimitive_type=mc.CDPrimitiveType.CAPSULE)
    bunnycm.set_rgba([0.7, 0.7, 0.0, .2])
    bunnycm.show_local_frame()
    bunnycm.attach_to(base)
    bunnycm.change_cdmesh_type(mc.CDMeshType.CYLINDER)
    # bunnycm.show_cdmesh()
    bunnycm.show_cdprimitive()
    # bunnycm2 = CollisionModel(bunnycm)
    # rotmat = rm.rotmat_from_axangle(np.array([1, 0, 0]), math.pi / 2.0)
    # bunnycm2.set_rotmat(rotmat)
    # bunnycm2.unshow_cdprimitive()
    # bunnycm2.change_cdmesh_type("cylinder")
    # bunnycm2.show_cdmesh()
    # bunnycm2.attach_to(base)
    # base.run()

    bunnycm1 = CollisionModel(objpath, cdprimitive_type=mc.CDPrimitiveType.CYLINDER)
    bunnycm1.set_rgba([0.7, 0, 0.7, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, .01, 0]))
    bunnycm1.set_rotmat(rotmat)
    bunnycm1.attach_to(base)
    bunnycm1.show_cdprimitive()

    # bunnycm2 = bunnycm1.copy()
    # bunnycm2.change_cdprimitive_type(cdprimitive_type="surface_balls")
    # bunnycm2.set_rgba([0, 0.7, 0.7, 1.0])
    # rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    # bunnycm2.set_pos(np.array([0, .2, 0]))
    # bunnycm2.set_rotmat(rotmat)
    # bunnycm2.attach_to(base)

    # bunnycmpoints = bunnycm.sample_surface()
    # bunnycm1points = bunnycm1.sample_surface()
    # bunnycm2points = bunnycm2.sample_surface()
    # bpcm = gm.GeometricModel(bunnycmpoints)
    # bpcm1 = gm.GeometricModel(bunnycm1points)
    # bpcm2 = gm.GeometricModel(bunnycm2points)
    # bpcm.attach_to(base)
    # bpcm1.attach_to(base)
    # bpcm2.attach_to(base)
    # bunnycm2.show_cdmesh(end_type="box")
    # bunnycm.show_cdmesh(end_type="box")
    # bunnycm1.show_cdmesh(end_type="convexhull")
    tic = time.time()
    result, contacts = bunnycm.is_pcdwith(bunnycm1, toggle_contacts=True)
    toc = time.time()
    print("mesh cd cost: ", toc - tic)
    print(result)
    ct_objcm = mgm.GeometricModel(contacts)
    ct_objcm.attach_to(base)
    # tic = time.time()
    # bunnycm2.is_pcdwith([bunnycm, bunnycm1])
    # toc = time.time()
    # print("primitive cd cost: ", toc - tic)

    # gen_box().attach_to(base)
    base.run()
