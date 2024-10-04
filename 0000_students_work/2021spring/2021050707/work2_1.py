import numpy as np
from wrs import modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7, .05, .3]), lookat_pos=np.zeros(3))
    #衝突モデルの定義
    """
    object_ref = mcm.CollisionModel(initializer="./objects/bunnysim.stl", cdprim_type="box", cdmesh_type="triangles")
    object_ref.set_rgba([.9, .75, .35, 1])
    """
    object_ref1 = cm.CollisionModel(initor="./objects/Coffee_cup.stl", cdprim_type="box", cdmesh_type="triangles")
    object_ref1.set_rgba([.9, .75, .35, 1])
    object_ref2 = cm.CollisionModel(initor="./objects/Glass.stl", cdprim_type="box", cdmesh_type="triangles")
    object_ref2.set_rgba([.9, .75, .35, 1])

    """
    # ウサギ１
    object1 = object_ref.copy()
    object1.set_pos(np.array([0, -.08, 0]))
    # ウサギ２
    object2 = object_ref.copy()
    object2.set_pos(np.array([0, -.05, 0]))
    object2.change_cdprimitive_type(cdprim_type="surface_balls")

    #ウサギ１とウサギ２の衝突を検出
    pcd_result = object1.is_pcdwith(object2)
    #ウサギ１と２の当たり判定(primit)を表示
    object1.attach_to(base)
    object1.show_cdprimit()
    object2.attach_to(base)
    object2.show_cdprimit()
    #衝突を表示
    print(pcd_result)
    """

    """
    # ウサギ３
    object3 = object_ref.copy()
    object3.set_pos(np.array([0, .05, 0]))
    object3.change_cdmesh_type(cdmesh_type="convex_hull")
    #ウサギ４
    object4 = object_ref.copy()
    object4.set_pos(np.array([0, .08, 0]))

    #ウサギ３と４の衝突検出
    mcd_result, cd_points = object3.is_mcdwith(object4, toggle_contacts=True)
    #衝突点を表示
    for pnt in cd_points:
        mgm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], major_radius=.002).attach_to(base)
    #衝突結果を表示
    print(mcd_result)

    # ウサギ３と４の当たり判定(primit)を表示
    object3.attach_to(base)
    object3.show_cdmesh()
    object4.attach_to(base)
    object4.show_cdmesh()
    """

    # コーヒーカップ
    object5 = object_ref1.copy()
    object5.set_pos(np.array([0, 0, 0]))
    object5.change_cdmesh_type(cdmesh_type="convex_hull")
    # グラス
    object6 = object_ref2.copy()
    object6.set_pos(np.array([0, .06, 0]))
    object6.change_cdmesh_type(cdmesh_type="convex_hull")

    # 衝突検出
    mcd_result, cd_points = object5.is_mcdwith(object6, toggle_contacts=True)
    # 衝突点を表示
    for pnt in cd_points:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    # 衝突結果を表示
    print(mcd_result)

    # 当たり判定(primit)を表示
    object5.attach_to(base)
    object5.show_cdmesh()
    object6.attach_to(base)
    object6.show_cdmesh()

    base.run()
