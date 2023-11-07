import numpy as np
import modeling.collision_model as cm
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7,.05,.3]), lookat_pos=np.zeros(3))
    # ウサギのモデルのファイルを用いてCollisionModelを初期化します
    # ウサギ1~5はこのCollisionModelのコピーとして定義します
    object_ref = cm.CollisionModel(initor="./objects/bunnysim.stl",
                                   cdprimitive_type="box",
                                   cdmesh_type="default")
    object_ref.set_rgba([.9, .75, .35, 1])
    # ウサギ1
    object1 = object_ref.copy()
    object1.set_pos(np.array([0, -.19, 0]))
    # ウサギ2
    object2 = object_ref.copy()
    object2.set_pos(np.array([0, -.095, 0]))
    # ウサギ3　衝突検出用のprimitiveを表面にサンプリングした球形状のsurface_ballsへ変更します
    object3 = object_ref.copy()
    object3.change_cdprimitive_type(cdprimitive_type="surface_balls")
    object3.set_pos(np.array([0, .01, 0]))
    # ウサギ4
    object4 = object_ref.copy()
    object4.set_pos(np.array([0, .095, 0]))
    # ウサギ5　衝突検出用のmeshを凸包convex_hullへ変更します
    object5 = object_ref.copy()
    object5.change_cdmesh_type(cdmesh_type="convex_hull")
    object5.set_pos(np.array([0, .17, 0]))
    # ウサギ1の画面表示．元のモデルのみ書き出します
    object1.attach_to(base)
    # ウサギ2の画面表示．元のモデル上に，デフォルトのprimitive(box)も表示します
    object2.attach_to(base)
    object2.show_cdprimitive()
    # ウサギ3の画面表示．元のモデル上に，新たに設定したprimitive(surface_balls)も表示します
    object3.attach_to(base)
    object3.show_cdprimitive()
    # ウサギ4の画面表示．元のモデル上に，デフォルトのmeshも表示します
    object4.attach_to(base)
    object4.show_cdmesh()
    # ウサギ5の画面表示．元のモデル上に，新たに設定したmesh(convex_hull)も表示します
    object5.attach_to(base)
    object5.show_cdmesh()
    base.run()