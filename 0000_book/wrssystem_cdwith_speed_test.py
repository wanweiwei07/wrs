import numpy as np
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    import time
    from wrs import basis as rm, modeling as gm, modeling as cm
    import math

    base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))
    # object
    object_ref = cm.CollisionModel(initor="./objects/bunnysim.stl",
                                   cdprim_type=cm.CDPrimitiveType.BOX,
                                   cdmesh_type=cm.CDMeshType.OBB)

    # object 1
    object1 = object_ref.copy()
    # object1.change_cdmesh_type(cdmesh_type="convex_hull")
    object1.set_pos(np.array([0, .04, 0]))
    # object 2
    object2 = object_ref.copy()
    object2.set_pos(np.array([0, .07, 0]))
    #
    object1.attach_to(base)
    object1.show_cdmesh()
    object2.attach_to(base)
    object2.show_cdmesh()
    #
    tic = time.time()
    for i in range(100):
        mcd_result, cd_points = object1.is_mcdwith(object2, toggle_contacts=True)
    toc = time.time()
    print(f"box mesh time: {toc-tic}s")
    print(mcd_result)
    for pnt in cd_points:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    #
    tic = time.time()
    for i in range(100):
        mcd_result = object1.is_pcdwith(object2)
    toc = time.time()
    print(f"primitive time: {toc-tic}s")
    print(mcd_result)
    #
    object2.change_cdmesh_type(cdmesh_type=cm.CDMeshType.DEFAULT)
    object2.set_rotmat(rm.rotmat_from_axangle([0,0,1], -math.pi/6))
    tic = time.time()
    for i in range(100):
        mcd_result, cd_points = object1.is_mcdwith(object2, toggle_contacts=True)
    toc = time.time()
    print(f"triangle mesh time: {toc-tic}s")
    for pnt in cd_points:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    print(mcd_result)

    # object_ref_c = cmc.CollisionModel(initializer="./objects/bunnysim.stl",
    #                                   cdprim_type=mcm.CDPType.BOX,
    #                                   cdmesh_type=mcm.CDMType.DEFAULT)
    # # object 3
    # object3 = object_ref_c.copy()
    # # object1.change_cdmesh_type(cdmesh_type="convex_hull")
    # object3.set_pos(np.array([0, .14, 0]))
    # # object 4
    # object4 = object_ref.copy()
    # object4.set_pos(np.array([0, .17, 0]))
    # #
    # object3.attach_to(base)
    # object3.show_cdmesh()
    # object4.attach_to(base)
    # object4.show_cdmesh()
    # tic = time.time()
    # for i in range(100):
    #     mcd_result, cd_points = object3.is_mcdwith(object4, toggle_contacts=True)
    # toc = time.time()
    # print(f"old time: {toc-tic}s")
    # for pnt in cd_points:
    #     mgm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    # print(mcd_result)


    base.run()
