import copy
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, NodePath, BitMask32
from visualization.panda.world import ShowBase
import basis.dataadapter as da
import modeling.geometricmodel as gm
import modeling.modelcollection as mc
import modeling._pcdhelper as pcd
import modeling._mcdhelper as mcd


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

    def __init__(self, initor, btransparency=True, cdprimitive_type="box", expand_radius=None, name="defaultname",
                 userdefined_cdprimitive_fn=None):
        """
        :param initor:
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
        if isinstance(initor, CollisionModel):
            self._name = copy.deepcopy(initor.name)
            self._objpath = copy.deepcopy(initor.objpath)
            self._trimesh = copy.deepcopy(initor.trimesh)
            self._pdnp = copy.deepcopy(initor.pdnp)
            self._localframe = copy.deepcopy(initor.localframe)
            self._cdprimitive_type = copy.deepcopy(initor.cdprimitive_type)
        else:
            if cdprimitive_type is not None and cdprimitive_type not in ["box", "ball", "cylinder", "pointcloud", "userdefined"]:
                raise Exception("Wrong Collision Model type name.")
            super().__init__(initor=initor, btransparency=btransparency, name=name)
            self._cdprimitive_type = cdprimitive_type
            if cdprimitive_type is not None:
                if cdprimitive_type == "ball":
                    if expand_radius is None:
                        expand_radius = 0.015
                    cdnd = self._gen_surfaceballs_cdnp(name="cdnp_ball", radius=expand_radius)
                else:
                    if expand_radius is None:
                        expand_radius = 0.002
                    if cdprimitive_type == "box":
                        cdnd = self._gen_box_cdnp(name="cdnp_box", radius=expand_radius)
                    if cdprimitive_type == "cylinder":
                        cdnd = self._gen_cylindrical_cdnp(name="cdnp_cylinder", radius=expand_radius)
                    if cdprimitive_type == "pointcloud":
                        cdnd = self._gen_pointcloud_cdnp(name="cdnp_pointcloud", radius=expand_radius)
                    if cdprimitive_type == "userdefined":
                        cdnd = userdefined_cdprimitive_fn(name="cdnp_userdefined", radius=expand_radius)
                # use pdnp.getChild instead of a new self._cdnp variable as collision nodepath is not compatible with deepcopy
                self._pdnp.attachNewNode(cdnd)
                self._pdnp.getChild(1).setCollideMask(BitMask32(2**31))
                # self._cdnp = self._pdnp.attachNewNode(cdnd)
                # self._cdnp.node().setCollideMask(BitMask32(2**31))
            self._localframe = None

    @property
    def cdprimitive_type(self):
        return self._cdprimitive_type

    @property
    def cdnp(self):
        return self._pdnp.getChild(1) # child-0 = pdnp_raw, child-1 = cdnp

    def _gen_box_cdnp(self, name='cdnp_box', radius=0.01):
        """
        :param obstacle:
        :return:
        author: weiwei
        date: 20180811
        """
        if self._pdnp is None:
            raise ValueError("The defined object must has a nodepath!")
        bottom_left, top_right = self.pdnp_raw.getTightBounds()
        center = (bottom_left + top_right) / 2.0
        # enlarge the bounding box
        bottom_left -= (bottom_left - center).normalize() * radius
        top_right += (top_right - center).normalize() * radius
        collision_primitive = CollisionBox(bottom_left, top_right)
        collision_node = CollisionNode(name)
        collision_node.addSolid(collision_primitive)
        return collision_node

    def _gen_cylindrical_cdnp(self, name='cdnp_cylinder', radius=0.01):
        """
        :param trimeshmodel:
        :param name:
        :param radius:
        :return:
        author: weiwei
        date: 20200108
        """
        if self._pdnp is None:
            raise ValueError("The defined object must has a nodepath!")
        bottom_left, top_right = self.pdnp_raw.getTightBounds()
        center = (bottom_left + top_right) / 2.0
        # enlarge the bounding box
        bottomleft_adjustvec = bottom_left - center
        bottomleft_adjustvec[2] = 0
        bottomleft_adjustvec.normalize()
        bottom_left += bottomleft_adjustvec * radius
        topright_adjustvec = top_right - center
        topright_adjustvec[2] = 0
        topright_adjustvec.normalize()
        top_right += topright_adjustvec * radius
        bottomleft_pos = da.pdv3_to_npv3(bottom_left)
        topright_pos = da.pdv3_to_npv3(top_right)
        collision_node = CollisionNode(name)
        for angle in np.nditer(np.linspace(math.pi / 10, math.pi * 4 / 10, 4)):
            ca = math.cos(angle)
            sa = math.sin(angle)
            new_bottomleft_pos = np.array([bottomleft_pos[0] * ca, bottomleft_pos[1] * sa, bottomleft_pos[2]])
            new_topright_pos = np.array([topright_pos[0] * ca, topright_pos[1] * sa, topright_pos[2]])
            new_bottomleft = da.npv3_to_pdv3(new_bottomleft_pos)
            new_topright = da.npv3_to_pdv3(new_topright_pos)
            collision_primitive = CollisionBox(new_bottomleft, new_topright)
            collision_node.addSolid(collision_primitive)
        return collision_node

    def _gen_surfaceballs_cdnp(self, name='cdnp_ball', radius=0.01):
        """
        :param obstacle:
        :return:
        author: weiwei
        date: 20180811
        """
        if self._trimesh is None:
            raise ValueError("The defined object must has a trimesh!")
        nsample = int(math.ceil(self._trimesh.area / (radius * 0.3) ** 2))
        nsample = 120 if nsample > 120 else nsample  # threshhold
        samples = self._trimesh.sample_surface_even(self._trimesh, nsample)
        collision_node = CollisionNode(name)
        for sglsample in samples:
            collision_node.addSolid(CollisionSphere(sglsample[0], sglsample[1], sglsample[2], radius=radius))
        return collision_node

    def _gen_pointcloud_cdnp(self, name='cdnp_pointcloud', radius=0.02):
        """
        :param obstacle:
        :return:
        author: weiwei
        date: 20191210
        """
        if self._trimesh is None:
            raise ValueError("The defined object must has a trimesh!")
        collision_node = CollisionNode(name)
        for sglpnt in self._trimesh.vertices:
            collision_node.addSolid(CollisionSphere(sglpnt[0], sglpnt[1], sglpnt[2], radius=radius))
        return collision_node

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
            returnnp.setMat(self._pdnp.getMat())
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
            self._pdnp.reparentTo(obj.render)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_cm(self)
        else:
            print("Must be ShowBase, modeling.StaticGeometricModel, GeometricModel, CollisionModel, "
                  "or CollisionModelCollection!")

    def detach(self):
        # TODO detach from model collection?
        self._pdnp.detachNode()

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
            return mcd.is_box2box_collided(self, objcm_list)

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
    # bunnycm.attach_to(base)
    bunnycm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnycm.set_rotmat(rotmat)

    bunnycm1 = CollisionModel(objpath, cdprimitive_type="cylinder")
    bunnycm1.set_rgba([0.7, 0, 0.7, 1.0])
    bunnycm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, .01, 0]))
    bunnycm1.set_rotmat(rotmat)

    bunnycm2 = bunnycm1.copy()
    bunnycm2.set_rgba([0, 0.7, 0.7, 1.0])
    # bunnycm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnycm2.set_pos(np.array([0, .2, 0]))
    bunnycm2.set_rotmat(rotmat)

    bunnycmpoints, _ = bunnycm.sample_surface()
    bunnycm1points, _ = bunnycm1.sample_surface()
    bunnycm2points, _ = bunnycm2.sample_surface()
    bpcm = CollisionModel(bunnycmpoints, expand_radius=.01)
    bpcm1 = CollisionModel(bunnycm1points, expand_radius=.01)
    bpcm2 = CollisionModel(bunnycm2points, expand_radius=.01)
    # bpcm.attach_to(base)
    # bpcm1.attach_to(base)
    # bpcm2.attach_to(base)
    # bunnycm2.show_cdmesh(type='box')
    # bunnycm.show_cdmesh(type='box')
    bunnycm1.show_cdmesh(type='convexhull')
    tic = time.time()
    bunnycm2.is_mcdwith([bunnycm, bunnycm1])
    toc = time.time()
    print("mesh cd cost: ", toc - tic)
    tic = time.time()
    bunnycm2.is_pcdwith([bunnycm, bunnycm1])
    toc = time.time()
    print("primitive cd cost: ", toc - tic)
    # bunnycm2.show_cdprimit()
    # bunnycm.show_cdprimit()
    # bunnycm1.show_cdprimit()

    # gen_box().attach_to(base)
    base.run()