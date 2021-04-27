import numpy as np
import modeling.collision_model as cm
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))
    # object
    object_ref = cm.CollisionModel(initor="./objects/bunnysim.stl",
                                   cdprimit_type="box",
                                   cdmesh_type="triangles")
    object_ref.set_rgba([.9, .75, .35, 1])
    # object 1
    object1 = object_ref.copy()
    object1.set_pos(np.array([0, -.18, 0]))
    # object 2
    object2 = object_ref.copy()
    object2.set_pos(np.array([0, -.09, 0]))
    # object 3
    object3 = object_ref.copy()
    object3.change_cdprimitive_type(cdprimitive_type="surface_balls")
    object3.set_pos(np.array([0, .0, 0]))
    # object 4
    object4 = object_ref.copy()
    object4.set_pos(np.array([0, .09, 0]))
    # object 5
    object5 = object_ref.copy()
    object5.change_cdmesh_type(cdmesh_type="convex_hull")
    object5.set_pos(np.array([0, .18, 0]))
    # object 1 show
    object1.attach_to(base)
    # object 2 show
    object2.attach_to(base)
    object2.show_cdprimit()
    # object 3 show
    object3.attach_to(base)
    object3.show_cdprimit()
    # object 4 show
    object4.attach_to(base)
    object4.show_cdmesh()
    # object 5 show
    object5.attach_to(base)
    object5.show_cdmesh()
    base.run()
