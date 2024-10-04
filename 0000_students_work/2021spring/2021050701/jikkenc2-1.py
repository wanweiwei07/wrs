import math
import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, modeling as gm, modeling as cm

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([-.8, .3, .4]),lookat_pos=np.array([0, 0, .1]))
    # マグカップの生成
    object1 = cm.CollisionModel(initor="./objects/mug.stl",
                                cdprim_type="box",
                                cdmesh_type="obb")
    object1.set_pos(np.array([.04, 0, 0]))
    object1.set_rgba([.9, .75, .35, 1])
    object1.attach_to(base)
    object1.show_cdmesh()

    # 牛乳パックの生成
    object2 = cm.CollisionModel(initor="./objects/milkCarton.stl",
                                cdprim_type="box",
                                cdmesh_type="obb")
    object2.set_pos(np.array([0, .0, 0]))
    object2.set_rotmat(rm.rotmat_from_axangle(np.array([0, 0, 1]), math.pi / 2))
    object2.set_rgba([.9, .75, .35, 1])
    object2.attach_to(base)
    object2.show_cdmesh()

    # 衝突検出と表示
    mcd_result, mcd_points = object1.is_mcdwith(object2, toggle_contacts=True)
    for point in mcd_points:
        gm.gen_sphere(pos=point, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)

    base.run()