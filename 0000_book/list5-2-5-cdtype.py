import numpy as np
from wrs import modeling as mcm
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))
    object_ref = mcm.CollisionModel(initor="./objects/bunnysim.stl",
                                    cdprim_type=mcm.mc.CDPrimType.AABB,
                                    cdmesh_type=mcm.mc.CDMeshType.DEFAULT)
    object_ref.rgba = np.array([.9, .75, .35, 1])
    # object 1
    object1 = object_ref.copy()
    object1.pos = np.array([0, -.18, 0])
    # object 2
    object2 = object_ref.copy()
    object2.pos = np.array([0, -.09, 0])
    # object 3
    object3 = object_ref.copy()
    object3.change_cdprim_type(cdprim_type=mcm.mc.CDPrimType.SURFACE_BALLS, ex_radius=.01)
    object3.pos = np.array([0, .0, 0])
    # object 4
    object4 = object_ref.copy()
    object4.pos = np.array([0, .09, 0])
    # object 5
    object5 = object_ref.copy()
    object5.change_cdmesh_type(cdmesh_type=mcm.mc.CDMeshType.CONVEX_HULL)
    object5.pos = np.array([0, .18, 0])
    # object 1 show
    object1.attach_to(base)
    # object 2 show
    object2.attach_to(base)
    object2.show_cdprim()
    # object 3 show
    object3.attach_to(base)
    object3.show_cdprim()
    # object 4 show
    object4.attach_to(base)
    object4.show_cdmesh()
    # object 5 show
    object5.attach_to(base)
    object5.show_cdmesh()
    base.run()
