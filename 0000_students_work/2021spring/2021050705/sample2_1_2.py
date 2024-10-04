import numpy as np
from wrs import modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.7, .5, .3]), lookat_pos=np.zeros(3))

    #カップ1(緑)の生成
    object1 = cm.CollisionModel(initor="./objects/01cup.stl", cdprim_type="box", cdmesh_type="convex_hull")
    object1.set_rgba([0, 1, 0, 1])
    object1.set_pos(np.array([0, 0, 0]))
    #カップ2(青)の生成
    object2 = cm.CollisionModel(initor="./objects/02cup.stl", cdprim_type="box", cdmesh_type="convex_hull")
    object2.set_rgba([0, 0, 1, 1])
    object2.set_pos(np.array([0, .08, 0]))
    #カップ1と2のmesh間の衝突を検出します. ここで, 引数toggle_contactsをtrueにして, 衝突した点も呼び出し側に戻させます
    mcd_result, cd_points = object1.is_mcdwith(object2, toggle_contacts=True)
    #検出した衝突点を画面に表示させます
    for pnt in cd_points:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    #衝突の結果を出力します
    print(mcd_result)
    #カップ1と2とそのmeshを画面に表示します
    object1.attach_to(base)
    object1.show_cdmesh()
    object2.attach_to(base)
    object2.show_cdmesh()

    base.run()

