import copy
import math
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, NodePath, BitMask32
from visualization.panda.world import ShowBase
import basis.dataadapter as da
import modeling.geometricmodel as gm
import modeling.modelcollection as mc
import modeling._pcdhelper as pcd
import modeling._mcdhelper as mcd
import modeling._gimpact_cdhelper as gcd


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

    def __init__(self, initiator, btransparency=True, cdprimitive_type="box", expand_radius=None, name="auto",
                 userdefined_cdprimitive_fn=None):
        """
        :param initiator:
        :param btransparency:
        :param cdprimitive_type: box, ball, cylinder, pointcloud, userdefined
        :param expand_radius:
        :param name:
        :param userdefined_cdprimitive_fn: the collision primitive will be defined in the provided function
                                           if cdprimitive_type = external;
                                           protocal for the callback function: return CollisionNode,
                                           may have multiple CollisionSolid
        date: 201290312, 20201212
        """
        if isinstance(initiator, CollisionModel):
            self._name = copy.deepcopy(initiator.name)
            self._objpath = copy.deepcopy(initiator.objpath)
            self._objtrm = copy.deepcopy(initiator.objtrm)
            self._objpdnp = copy.deepcopy(initiator.objpdnp)
            self._localframe = copy.deepcopy(initiator.localframe)
            self._cdprimitive_type = copy.deepcopy(initiator.cdprimitive_type)
        else:
            if cdprimitive_type is not None and cdprimitive_type not in ["box",
                                                                         "surface_ball",
                                                                         "cylinder",
                                                                         "point_cloud",
                                                                         "user_defined"]:
                raise Exception("Wrong Collision Model type name.")
            super().__init__(initiator=initiator, btransparency=btransparency, name=name)
            self._cdprimitive_type = cdprimitive_type
            if cdprimitive_type is not None:
                collision_node = self._update_cdprimit(cdprimitive_type, expand_radius, userdefined_cdprimitive_fn)
                # use pdnp.getChild instead of a new self._cdnp variable as collision nodepath is not compatible with deepcopy
                self._objpdnp.attachNewNode(collision_node)
                self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))
            self._localframe = None

    def _update_cdprimit(self, cdprimitive_type, expand_radius, userdefined_cdprimitive_fn):
        if cdprimitive_type == 'surface_ball':
            if expand_radius is None:
                expand_radius = 0.015
            collision_node = pcd.gen_surfaceballs_cdnp(self.objtrm, name='cdnp_surface_ball', radius=expand_radius)
        else:
            if expand_radius is None:
                expand_radius = 0.002
            if cdprimitive_type == "box":
                collision_node = pcd.gen_box_cdnp(self.objpdnp_raw, name='cdnp_ball', radius=expand_radius)
            if cdprimitive_type == "cylinder":
                collision_node = pcd.gen_cylindrical_cdnp(self.objpdnp_raw, name='cdnp_cyl', radius=expand_radius)
            if cdprimitive_type == "point_cloud":
                collision_node = pcd.gen_pointcloud_cdnp(self.objtrm, name='cdnp_ptc', radius=expand_radius)
            if cdprimitive_type == "user_defined":
                collision_node = userdefined_cdprimitive_fn(name="cdnp_usrdef", radius=expand_radius)
        return collision_node

    @property
    def cdprimitive_type(self):
        return self._cdprimitive_type

    @property
    def cdnp(self):
        return self._objpdnp.getChild(1) # child-0 = pdnp_raw, child-1 = cdnp

    def change_cdprimit(self, cdprimitive_type='ball', expand_radius=.01, userdefined_cdprimitive_fn=None):
        """
        :param cdprimitive_type:
        :param expand_radius:
        :param userdefined_cdprimitive_fn: None, only used when cdprimitive_type == 'userdefined'
        :return:
        author: weiwei
        date: 20210116
        """
        cdnd = self._update_cdprimit(cdprimitive_type, expand_radius, userdefined_cdprimitive_fn)
        # use pdnp.getChild instead of a new self._cdnp variable as collision nodepath is not compatible with deepcopy
        self.cdnp.removeNode()
        self._objpdnp.attachNewNode(cdnd)
        self._objpdnp.getChild(1).setCollideMask(BitMask32(2 ** 31))

    def copy_cdnp_to(self, nodepath, homomat=None, clearmask = False):
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
            returnnp.setMat(da.npmat4_to_pdmat4(homomat))
        return returnnp

    def is_pcdwith(self, objcm):
        """
        Is the primitives of this cm collide with the primitives of the given cm
        :param objcm: one or a list of Collision Model object
        author: weiwei
        date: 20201116
        """
        return pcd.is_cdprimit2cdprimit_collided(self, objcm)

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

    def is_mcdwith(self, objcm_list, type='triangles2triangles'):
        """
        Is the mesh of the cm collide with the mesh of the given cm
        :param objcm_list: one or a list of Collision Model object
        :param type: 'triangles2triangles', 'box2triangles', 'box2box'
        author: weiwei
        date: 20201116
        """
        if type == 'triangles2triangles':
            return mcd.is_triangles2triangles_collided(self, objcm_list)
        if type == 'box2triangles':
            return mcd.is_box2triangles_collided(self, objcm_list)
        if type == 'box2box':
            return mcd.is_box2box_collided(self, objcm_list)
        if type == 'convexhull2triangles':
            return mcd.is_convexhull2triangles_collided(self, objcm_list)

    def show_cdmesh(self, type='triangles'):
        self. unshow_cdmesh()
        if type == 'triangles':
            self._bullnode = mcd.show_triangles_cdmesh(self)
        elif type == 'convexhull':
            self._bullnode = mcd.show_convexhull_cdmesh(self)
        elif type == 'box':
            self._bullnode = mcd.show_box_cdmesh(self)
        else:
            raise NotImplementedError('The requested '+type+' type cdmesh is not supported!')

    def unshow_cdmesh(self):
        if hasattr(self, '_bullnode'):
            mcd.unshow(self._bullnode)

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


if __name__ == "__main__":
    import os
    import math
    import time
    import numpy as np
    import basis
    import basis.robotmath as rm
    import visualization.panda.world as wd

    base = wd.World(campos=[.3, .3, .3], lookatpos=[0, 0, 0], toggledebug=True)
    objpath = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    bunnycm = CollisionModel(objpath)
    bunnycm.set_rgba([0.7, 0.7, 0.0, 1.0])
    bunnycm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnycm.set_rotmat(rotmat)

    bunnycm1 = CollisionModel(objpath, cdprimitive_type="cylinder")
    bunnycm1.set_rgba([0.7, 0, 0.7, 1.0])
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, .01, 0]))
    bunnycm1.set_rotmat(rotmat)

    bunnycm2 = bunnycm1.copy()
    bunnycm2.change_cdprimit(cdprimitive_type='surface_ball')
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