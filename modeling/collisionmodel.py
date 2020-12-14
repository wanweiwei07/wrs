import copy
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, CollisionSphere, NodePath
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

    def __init__(self, objinit, btransparency=True, cdprimitive_type="box", expand_radius=None, name="defaultname",
                 external_cdprimitive_callback=None):
        """
        :param objinit:
        :param btransparency:
        :param cdprimitive_type: box, ball, cylinder, pointcloud, external
        :param expand_radius:
        :param name:
        :param external_cdprimitive_callback: the collision primitive will be defined in the provided callback function
                                              if cdprimitive_type = external;
                                              protocal for the callback function: return CollisionNode,
                                              may have multiple CollisionSolid
        date: 201290312, 20201212
        """
        if isinstance(objinit, CollisionModel):
            self._name = objinit.name
            self._objpath = copy.deepcopy(objinit.objpath)
            self._trimesh = copy.deepcopy(objinit.trimesh)
            self._pdnp = copy.deepcopy(objinit.pdnp)
            self._pdnp_raw = self._pdnp.find(self.name + "_raw")
            self._localframe = copy.deepcopy(objinit.localframe)
            self._cdnp = objinit.copy_cdnp_to(self._pdnp)
            self._cdprimitive_type = objinit.cdprimitive_type
            self._pcd = objinit.pcd  # primitive collision detector
            self._mcd = objinit.mcd  # bullet collision detector
        else:
            if cdprimitive_type is not None and cdprimitive_type not in ["box", "ball", "cylinder", "pointcloud"]:
                raise Exception("Wrong Collision Model type name.")
            super().__init__(objinit=objinit, btransparency=btransparency, name=name)
            self._cdprimitive_type = cdprimitive_type
            if cdprimitive_type is not None:
                if cdprimitive_type == "ball":
                    if expand_radius is None:
                        expand_radius = 0.015
                    cdnd = self._gen_surfaceballs_cdnp(name=self.name+"_ballcd", radius=expand_radius)
                else:
                    if expand_radius is None:
                        expand_radius = 0.002
                    if cdprimitive_type == "box":
                        cdnd = self._gen_box_cdnp(name=self.name+"_boxcd", radius=expand_radius)
                    if cdprimitive_type == "cylinder":
                        cdnd = self._gen_cylindrical_cdnp(name=self.name+"_cylindricalcd", radius=expand_radius)
                    if cdprimitive_type == "pointcloud":
                        cdnd = self._gen_pointcloud_cdnp(name=self.name+"_pointcloudcd", radius=expand_radius)
                    if cdprimitive_type == "external":
                        cdnd = external_cdprimitive_callback(cmobj=self, radius=expand_radius)
                self._cdnp = self._pdnp.attachNewNode(cdnd)
            self._localframe = None
            self._pcd = pcd  # primitive collision detector
            self._mcd = mcd  # bullet collision detector

    @property
    def cdprimitive_type(self):
        return self._cdprimitive_type

    @property
    def cdnp(self):
        return self._cdnp

    @property
    def pcd(self):
        return self._pcd

    @property
    def mcd(self):
        return self._mcd

    def _gen_box_cdnp(self, name='boxbound', radius=0.01):
        """
        :param obstacle:
        :return:
        author: weiwei
        date: 20180811
        """
        if self._trimesh is None:
            raise ValueError("The defined object must has a nodepath!")
        # TODO? Use trimesh.bounds to get decoupled from panda3d (slower)
        bottom_left, top_right = self._pdnp.getTightBounds()
        center = (bottom_left + top_right) / 2.0
        # enlarge the bounding box
        bottom_left -= (bottom_left - center).normalize() * radius
        top_right += (top_right - center).normalize() * radius
        collision_primitive = CollisionBox(bottom_left, top_right)
        collision_node = CollisionNode(name)
        collision_node.addSolid(collision_primitive)
        return collision_node

    def _gen_cylindrical_cdnp(self, name='collisionbound', radius=0.01):
        """
        :param trimeshmodel:
        :param name:
        :param radius:
        :return:
        author: weiwei
        date: 20200108
        """
        if self._trimesh is None:
            raise ValueError("The defined object must has a nodepath!")
        # TODO? Use trimesh.bounds to get decoupled from panda3d (slower)
        bottom_left, top_right = self._pdnp.getTightBounds()
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

    def _gen_surfaceballs_cdnp(self, name='ballcd', radius=0.01):
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

    def _gen_pointcloud_cdnp(self, name='pointcloudcd', radius=0.02):
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

    def copy_cdnp_to(self, nodepath, homomat=None, clearmask = True):
        """
        Return a nodepath including the cdcn,
        the returned nodepath is attached to the given one
        :param nodepath: parent np
        :param homomat: allow specifying a special homomat to virtually represent a pose that is different from the mesh
        :return:
        author: weiwei
        date: 20180811
        """
        returnnp = nodepath.attachNewNode(copy.deepcopy(self._cdnp.getNode(0)))
        if clearmask:
            returnnp.node().setCollideMask(0x00)
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
        if isinstance(objcm, CollisionModel):
            return pcd.is_cmcm_collided(self, objcm)
        elif isinstance(objcm, list):
            return pcd.is_cmcmlist_collided(self, objcm)

    def attach_to(self, obj):
        if isinstance(obj, ShowBase):
            # for rendering to base.render
            self._pdnp.reparentTo(obj.render)
        elif isinstance(obj, mc.ModelCollection):
            obj.add_cm(self)
        else:
            print("Must be ShowBase, modeling.StaticGeometricModel, GeometricModel, CollisionModel, "
                  "or CollisionModelCollection!")

    def show_cdprimit(self):
        """
        Show collision node
        """
        self._cdnp.show()

    def unshow_cdprimit(self):
        self._cdnp.hide()

    def is_mcdwith(self, objcm):
        """
        Is the mesh of the cm collide with the mesh of the given cm
        :param objcm: one or a list of Collision Model object
        author: weiwei
        date: 20201116
        """
        if isinstance(objcm, CollisionModel):
            return self._mcd.is_mesh_cmcm_collided(self, objcm)
        elif isinstance(objcm, list):
            return self._mcd.is_mesh_cmcmlist_collided(self, objcm)

    def show_cdmesh(self):
        """
        :return:
        """
        self._bullnode = self._mcd.show_meshcm(self)

    def unshow_cdmesh(self):
        """
        :return:
        """
        if hasattr(self, '_bullnode'):
            self._mcd.unshow(self._bullnode)

    def is_mboxcdwith(self, objcm):
        raise NotImplementedError

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
    box_sgm = gm.gen_box(extent=extent, homomat=homomat, rgba=rgba)
    box_cm = CollisionModel(box_sgm)
    return box_cm


if __name__ == "__main__":
    import os
    import math
    import time
    import numpy as np
    import basis.robotmath as rm
    import visualization.panda.world as wd

    base = wd.World(campos=[.3, .3, .3], lookatpos=[0, 0, 0], toggledebug=True)
    this_dir, this_filename = os.path.split(__file__)
    objpath = os.path.join(this_dir, "objects", "bunnysim.stl")
    bunnycm = CollisionModel(objpath)
    bunnycm.set_color([0.7, 0.7, 0.0, 1.0])
    bunnycm.attach_to(base)
    bunnycm.show_localframe()
    rotmat = rm.rotmat_from_axangle([1, 0, 0], math.pi / 2.0)
    bunnycm.set_rotmat(rotmat)

    bunnycm1 = CollisionModel(objpath, cdprimitive_type="cylinder")
    bunnycm1.set_color([0.7, 0, 0.7, 1.0])
    bunnycm1.attach_to(base)
    rotmat = rm.rotmat_from_euler(0, 0, math.radians(15))
    bunnycm1.set_pos(np.array([0, .01, 0]))
    bunnycm1.set_rotmat(rotmat)

    bunnycm2 = CollisionModel(objpath, cdprimitive_type="cylinder")
    bunnycm2.set_color([0, 0.7, 0.7, 1.0])
    bunnycm2.attach_to(base)
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    bunnycm1.set_pos(np.array([0, .2, 0]))
    bunnycm1.set_rotmat(rotmat)

    bunnycmpoints, _ = bunnycm.sample_surface()
    bunnycm1points, _ = bunnycm1.sample_surface()
    bunnycm2points, _ = bunnycm2.sample_surface()
    bpcm = CollisionModel(bunnycmpoints, expand_radius=.01)
    bpcm1 = CollisionModel(bunnycm1points, expand_radius=.01)
    bpcm2 = CollisionModel(bunnycm2points, expand_radius=.01)
    bpcm.attach_to(base)
    bpcm1.attach_to(base)
    bpcm2.attach_to(base)
    tic = time.time()
    bunnycm2.is_mcdwith([bunnycm, bunnycm1])
    toc = time.time()
    print("mesh cd cost: ", toc - tic)
    tic = time.time()
    bunnycm2.is_pcdwith([bunnycm, bunnycm1])
    toc = time.time()
    print("primitive cd cost: ", toc - tic)
    bunnycm2.show_cdmesh()
    bunnycm.show_cdmesh()
    bunnycm1.show_cdmesh()
    bunnycm2.show_cdprimit()
    bunnycm.show_cdprimit()
    bunnycm1.show_cdprimit()

    gen_box().attach_to(base)
    base.run()