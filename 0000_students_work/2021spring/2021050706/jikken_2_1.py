import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import modeling.collision_model as cm

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.07, .05, .1]), lookat_pos=np.zeros(3))

    object1 = cm.CollisionModel(initor="./part_studio_1.stl", cdprimit_type="box", cdmesh_type="convex_hull")
    object1.set_rgba([.9, .75, .35, 1])
    object1.set_pos(np.array([0, -.08, 0]))

    object2 = cm.CollisionModel(initor="./part_studio_2.stl", cdprimit_type="box", cdmesh_type="convex_hull")
    object2.set_rgba([.3, .9, .6, 1])
    object2.set_pos(np.array([0, -.05, 0]))

    pcd_result, cd_points = object1.is_mcdwith(object2, toggle_contacts=True)

    for pnt in cd_points:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)

    object1.attach_to(base)
    object1.show_cdmesh()
    object2.attach_to(base)
    object2.show_cdmesh()

    base.run()

