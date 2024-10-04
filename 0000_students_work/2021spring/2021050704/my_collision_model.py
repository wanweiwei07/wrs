import math
import numpy as np
from wrs import basis as rm, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))

#オブジェクトの読み込み
object1_ref = cm.CollisionModel(initor="./my_objects/hand_spinner.stl",
                                cdprim_type="box",
                                cdmesh_type="triangles")
object2_ref = cm.CollisionModel(initor="./my_objects/LEGO.stl",
                                cdprim_type="box",
                                cdmesh_type="triangles")

#色の定義
object1_ref.set_rgba([0,1,0,1])
object2_ref.set_rgba([0,0,1,1])

#ハンドスピナーの定義
object1 = object1_ref.copy()
object1.change_cdmesh_type(cdmesh_type="convex_hull")
object1.set_pos(np.array([0,0,0]))
object1.set_rotmat(rm.rotmat_from_axangle(axis=np.array([1,0,0]),angle=math.pi/2))
object1.set_pos(np.array([0,.028,.02]))

#レゴの定義
object2 = object2_ref.copy()
object2.set_pos(np.array([0,.08,0]))

#衝突の判定
mcd_result, cd_points = object1.is_mcdwith(object2, toggle_contacts=True)
for pnt in cd_points:
    gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
print(mcd_result)

#オブジェクトの表示
object1.attach_to(base)
object1.show_cdmesh()
object2.attach_to(base)
object2.show_cdmesh()

base.run()

