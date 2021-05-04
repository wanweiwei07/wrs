import copy
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, NodePath, BitMask32
from visualization.panda.world import ShowBase
import basis.robot_math as rm
import basis.data_adapter as da
import modeling.geometric_model as gm
import modeling.model_collection as mc
import modeling._panda_cdhelper as pcd
import modeling._ode_cdhelper as mcd


# import modeling._gimpact_cdhelper as mcd
# import modeling._bullet_cdhelper as mcd

class CollisionModel(gm.GeometricModel):
    """
    Load an object as a collision model
    Both collison primitives will be generated automatically
    Note: This class heaviliy depends on Panda3D
          cdnp nodepath of collsion detection primitives
          pdnp nodepath of mesh+decorations; decorations = coordinate frames, markers, etc.
          pdnp nodepath of mesh
    author: weiwei
    date: 20190312
    """

    def __init__(self,
                 initor,
                 cdprimit_type='box',
                 cdmesh_type='triangles',
                 expand_radius=None,
                 name="auto",
                 userdefined_cdprimitive_fn=None,
                 btransparency=True,
                 btwosided=False):
        """
        :param initor:
        :param btransparency:
        :param cdprimit_type: box, ball, cylinder, point_cloud, user_defined
        :param cdmesh_type: aabb, obb, convex_hull, triangulation
        :param expand_radius:
        :param name:
        :param userdefined_cdprimitive_fn: the collision primitive will be defined in the provided function
                                           if cdprimitive_type = external;
                                           protocal for the callback function: return CollisionNode,
                                           may have multiple CollisionSolid
        date: 201290312, 20201212
        """
        if isinstance(initor, CollisionModel):
            self._name = copy.deepcopy(initor.name)
            self._objpath = copy.deepcopy(initor.objpath)
            self._objtrm = copy.deepcopy(initor.objtrm)
            self._objpdnp = copy.deepcopy(initor.objpdnp)
            self._localframe = copy.deepcopy(initor.localframe)
            self._cdprimitive_type = copy.deepcopy(initor.cdprimitive_type)
            self._cdmesh_type = copy.deepcopy(initor.cdmesh_type)
        else:
            super().__init__(initor=initor, name=name, btransparency=btransparency, btwosided=btwosided)
            self._cdprimitive_type, collision_node = self._update_cdprimit(cdprimit_type,
                                                                           expand_radius,
                                                                           userdefined_cdprimitive_fn)
            # use pdnp.getChild instead of a new self._cdnp variable as collision nodepath is not compatible with deepcopy
            self._objpdnp.attachNewNode(collision_node)
            self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))
            self.cdmesh_type = cdmesh_type
            self._localframe = None

    def _update_cdprimit(self, cdprimitive_type, expand_radius, userdefined_cdprimitive_fn):
        if cdprimitive_type is not None and cdprimitive_type not in ['box',
                                                                     'surface_balls',
                                                                     'cylinder',
                                                                     'polygons',
                                                                     'point_cloud',
                                                                     'user_defined']:
            raise ValueError("Wrong primitive collision model type name!")
        if cdprimitive_type == 'surface_balls':
            if expand_radius is None:
                expand_radius = 0.015
            collision_node = pcd.gen_surfaceballs_cdnp(self.objtrm, name='cdnp_surface_ball', radius=expand_radius)
        else:
            if expand_radius is None:
                expand_radius = 0.002
            if cdprimitive_type == "box":
                collision_node = pcd.gen_box_cdnp(self.objpdnp_raw, name='cdnp_box', radius=expand_radius)
            if cdprimitive_type == "cylinder":
                collision_node = pcd.gen_cylindrical_cdnp(self.objpdnp_raw, name='cdnp_cyl', radius=expand_radius)
            if cdprimitive_type == "polygons":
                collision_node = pcd.gen_polygons_cdnp(self.objpdnp_raw, name='cdnp_plys', radius=expand_radius)
            if cdprimitive_type == "point_cloud":
                collision_node = pcd.gen_pointcloud_cdnp(self.objtrm, name='cdnp_ptc', radius=expand_radius)
            if cdprimitive_type == "user_defined":
                collision_node = userdefined_cdprimitive_fn(name="cdnp_usrdef", radius=expand_radius)
        return cdprimitive_type, collision_node

    @property
    def cdprimitive_type(self):
        return self._cdprimitive_type

    @property
    def cdmesh_type(self):
        return self._cdmesh_type

    @cdmesh_type.setter
    def cdmesh_type(self, cdmesh_type):
        if cdmesh_type is not None and cdmesh_type not in ['aabb',
                                                           'obb',
                                                           'convex_hull',
                                                           'triangles']:
            raise ValueError("Wrong mesh collision model type name!")
        self._cdmesh_type = cdmesh_type

    @property
    def cdnp(self):
        return self._objpdnp.getChild(1)  # child-0 = pdnp_raw, child-1 = cdnp

    @property
    def cdmesh(self):
        return mcd.gen_cdmesh_vvnf(*self.extract_rotated_vvnf())

    def extract_rotated_vvnf(self):
        if self.cdmesh_type == 'aabb':
            objtrm = self.objtrm.bounding_box
        elif self.cdmesh_type == 'obb':
            objtrm = self.objtrm.bounding_box_oriented
        elif self.cdmesh_type == 'convex_hull':
            objtrm = self.objtrm.convex_hull
        elif self.cdmesh_type == 'triangles':
            objtrm = self.objtrm
        homomat = self.get_homomat()
        vertices = rm.homomat_transform_points(homomat, objtrm.vertices)
        vertex_normals = rm.homomat_transform_points(homomat, objtrm.vertex_normals)
        faces = objtrm.faces
        return vertices, vertex_normals, faces

    def change_cdprimitive_type(self, cdprimitive_type='ball', expand_radius=.01, userdefined_cdprimitive_fn=None):
        """
        :param cdprimitive_type:
        :param expand_radius:
        :param userdefined_cdprimitive_fn: None, only used when cdprimitive_type == 'userdefined'
        :return:
        author: weiwei
        date: 20210116
        """
        self._cdprimitive_type, cdnd = self._update_cdprimit(cdprimitive_type, expand_radius,
                                                             userdefined_cdprimitive_fn)
        # use _objpdnp.getChild instead of a new self._cdnp variable as collision nodepath is not compatible with deepcopy
        self.cdnp.removeNode()
        self._objpdnp.attachNewNode(cdnd)
        self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))

    def change_cdmesh_type(self, cdmesh_type='convex_hull'):
        """
        :param cdmesh_type:
        :return:
        author: weiwei
        date: 20210117
        """
        self.cdmesh_type = cdmesh_type

    def copy_cdnp_to(self, nodepath, homomat=None, clearmask=False):
        """
        Return a nodepath including the cdcn,
        the returned nodepath is attached to the given one
        :param nodepath: parent np
        :param homomat: allow specifying a special homomat to virtually represent a pose that is different from the mesh
        :return:
        author: weiwei
        date: 20180811
        """
        returnnp = nodepath.attachNewNode(copy.deepcopy(self.cdnp.getNode(0)))
        if clearmask:
            returnnp.node().setCollideMask(0x00)
        else:
            returnnp.node().setCollideMask(self.cdnp.getCollideMask())
        if homomat is None:
            returnnp.setMat(self._objpdnp.getMat())
        else:
            returnnp.setMat(da.npmat4_to_pdmat4(homomat))  # scale is reset to 1 1 1 after setMat to the given homomat
            returnnp.setScale(self._objpdnp.getScale())
        return returnnp

    def is_pcdwith(self, objcm):
        """
        Is the primitives of this cm collide with the primitives of the given cm
        :param objcm: one or a list of Collision Model object
        author: weiwei
        date: 20201116
        """
        return pcd.is_collided(self, objcm)

    def attach_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._objpdnp.reparentTo(obj.render)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_cm(self)
        else:
            print("Must be ShowBase, modeling.StaticGeometricModel, GeometricModel, CollisionModel, "
                  "or CollisionModelCollection!")

    def detach(self):
        # TODO detach from model collection?
        self._objpdnp.detachNode()

    def show_cdprimit(self):
        """
        Show collision node
        """
        self.cdnp.show()

    def unshow_cdprimit(self):
        self.cdnp.hide()

    def is_mcdwith(self, objcm_list, toggle_contacts=False):
        """
        Is the mesh of the cm collide with the mesh of the given cm
        :param objcm_list: one or a list of Collision Model object
        :param toggle_contacts: return a list of contact points if toggle_contacts is True
        author: weiwei
        date: 20201116
        """
        if not isinstance(objcm_list, list):
            objcm_list = [objcm_list]
        for objcm in objcm_list:
            iscollided, contact_points = mcd.is_collided(self, objcm)
            if iscollided and toggle_contacts:
                return [True, contact_points]
            elif iscollided:
                return True
        return [False, []] if toggle_contacts else False

    def ray_hit(self, point_from, point_to, option="all"):
        """
        check the intersection between segment point_from-point_to and the mesh
        :param point_from: 1x3 nparray
        :param point_to:
        :param option: "all" or “closest"
        :return:
        author: weiwei
        date: 20210504
        """
        if option == "all":
            contact_points, contact_normals = mcd.rayhit_all(point_from, point_to, self)
            return contact_points, contact_normals
        elif option == "closest":
            contact_point, contact_normal = mcd.rayhit_closet(point_from, point_to, self)
            return contact_point, contact_normal

    def show_cdmesh(self):
        vertices, vertex_normals, faces = self.extract_rotated_vvnf()
        objwm = gm.WireFrameModel(da.trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces))
        self._tmp_shown_cdmesh = objwm.attach_to(base)

    def unshow_cdmesh(self):
        if hasattr(self, '_tmp_shown_cdmesh'):
            self._tmp_shown_cdmesh.detach()

    def is_mboxcdwith(self, objcm):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)


def gen_box(extent=np.array([.1, .1, .1]), homomat=np.eye(4), rgba=np.array([1, 0, 0, 1])):
    """
    :param extent:
    :param homomat:
    :return:
    author: weiwei
    date: 20201202
    """
    box_sgm = gm.gen_box(extent=extent, homomat=homomat, rgba=rgba)
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
    sphere_sgm = gm.gen_sphere(pos=pos, radius=radius, rgba=rgba)
    sphere_cm = CollisionModel(sphere_sgm)
    return sphere_cm


def gen_stick(spos=np.array([.0, .0, .0]),
              epos=np.array([.0, .0, .1]),
              thickness=.005, type="rect",
              rgba=[1, 0, 0, 1],
              sections=8):
    """
    :param spos:
    :param epos:
    :param thickness:
    :param rgba:
    :return:
    author: weiwei
    date: 20210328
    """
    stick_sgm = gm.gen_stick(spos=spos, epos=epos, thickness=thickness, type=type, rgba=rgba, sections=sections)
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
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnycm = CollisionModel(objpath, cdprimit_type='polygons')
    bunnycm.set_rgba([0.7, 0.7, 0.0, .2])
    bunnycm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnycm.set_rotmat(rotmat)
    bunnycm.show_cdprimit()

    bunnycm1 = CollisionModel(objpath, cdprimit_type="cylinder")
    bunnycm1.set_rgba([0.7, 0, 0.7, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, .01, 0]))
    bunnycm1.set_rotmat(rotmat)

    bunnycm2 = bunnycm1.copy()
    bunnycm2.change_cdprimitive_type(cdprimitive_type='surface_balls')
    bunnycm2.set_rgba([0, 0.7, 0.7, 1.0])
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnycm2.set_pos(np.array([0, .2, 0]))
    bunnycm2.set_rotmat(rotmat)

    bunnycm.attach_to(base)
    bunnycm1.attach_to(base)
    bunnycm2.attach_to(base)
    bunnycm.show_cdprimit()
    bunnycm1.show_cdprimit()
    bunnycm2.show_cdprimit()

    bunnycmpoints, _ = bunnycm.sample_surface()
    bunnycm1points, _ = bunnycm1.sample_surface()
    bunnycm2points, _ = bunnycm2.sample_surface()
    bpcm = gm.GeometricModel(bunnycmpoints)
    bpcm1 = gm.GeometricModel(bunnycm1points)
    bpcm2 = gm.GeometricModel(bunnycm2points)
    bpcm.attach_to(base)
    bpcm1.attach_to(base)
    bpcm2.attach_to(base)
    # bunnycm2.show_cdmesh(type='box')
    # bunnycm.show_cdmesh(type='box')
    # bunnycm1.show_cdmesh(type='convexhull')
    # tic = time.time()
    # bunnycm2.is_mcdwith([bunnycm, bunnycm1])
    # toc = time.time()
    # print("mesh cd cost: ", toc - tic)
    # tic = time.time()
    # bunnycm2.is_pcdwith([bunnycm, bunnycm1])
    # toc = time.time()
    # print("primitive cd cost: ", toc - tic)

    # gen_box().attach_to(base)
    base.run()
