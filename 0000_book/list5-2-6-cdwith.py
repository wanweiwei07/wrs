import numpy as np
from wrs import modeling as mgm, modeling as mcm
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))
    # object
    object_ref = mcm.CollisionModel(initor="./objects/bunnysim.stl",
                                    cdprim_type=mcm.mc.CDPrimType.AABB,
                                    cdmesh_type=mcm.mc.CDMeshType.DEFAULT)
    object_ref.rgba = np.array([.9, .75, .35, 1])
    # object1
    object1 = object_ref.copy()
    object1.pos = np.array([0, -.07, 0])
    # object 2
    object2 = object_ref.copy()
    object2.pos = np.array([0, -.04, 0])
    object2.change_cdprim_type(cdprim_type=mcm.mc.CDPrimType.SURFACE_BALLS, ex_radius=.01)
    #
    object1.attach_to(base)
    object1.show_cdprim()
    object2.attach_to(base)
    object2.show_cdprim()
    pcd_result = object1.is_pcdwith(object2)
    print(pcd_result)

    # object 3
    object3 = object_ref.copy()
    object3.change_cdmesh_type(cdmesh_type=mcm.mc.CDMeshType.CONVEX_HULL)
    object3.pos = np.array([0, .04, 0])
    # object 4
    object4 = object_ref.copy()
    object4.pos = np.array([0, .07, 0])
    #
    object3.attach_to(base)
    object3.show_cdmesh()
    object4.attach_to(base)
    object4.show_cdmesh()
    mcd_result, cd_points = object3.is_mcdwith(object4, toggle_contacts=True)
    for pnt in cd_points:
        mgm.gen_sphere(pos=pnt, rgb=np.array([1, 0, 0]), alpha=1, radius=.002).attach_to(base)
    print(mcd_result)
    base.run()
