# kadai02_01.py
# 2021/04/30 実験C第二回の課題1
# glassと星形のモデルを作成し，glass1(黄色)とstar1(黄色)間，star1(黄色)とstar2(紫色)間，glass2(紫色)とstar2(紫色)間の干渉をチェックしました．
# 凸包の三角メッシュのものは黄色で，もともとの三角メッシュのものは紫色で表示しました．

import math
import numpy as np
from wrs.basis import robot_math as rm
from wrs import modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.6, .3, .8]), lookat_pos=np.zeros(3))
    gm.gen_frame(axis_length=.2, axis_radius=.01).attach_to(base)  # グローバル座標系

    # glassと星形のモデルのファイルを用いてCollisionModelを初期化します
    # glass1, star1はmesh(convex_hull)の設定で作成します
    glass1 = cm.CollisionModel(initor="objects/glass1.stl", cdprim_type="surface_balls", cdmesh_type="convex_hull")
    glass1.set_rgba([.9, .75, .35, 1])  # 黄色に変更
    glass1.set_pos(np.array([0, -.06, 0]))
    star1 = cm.CollisionModel(initor="objects/star2.stl", cdprim_type="surface_balls", cdmesh_type="convex_hull")
    star1.set_rgba([.9, .75, .35, 1])  # 黄色に変更
    star1.set_pos(np.array([0, .01, .07]))
    star1.set_rotmat(rm.rotmat_from_axangle(axis=[0, 0, 1], angle=math.pi/2.))

    # glass1, star1をそれぞれコピーし，mesh(triangles)に変更してglass2, star2を作成します
    glass2 = glass1.copy()
    glass2.change_cdmesh_type(cdmesh_type="triangles")
    glass2.set_pos(np.array([.02, .13, .025]))
    glass2.set_rgba([.75, .35, .9, 1])  # 紫色に変更
    star2 = star1.copy()
    star2.change_cdmesh_type(cdmesh_type="triangles")
    star2.set_pos(np.array([.01, .085, .1]))
    star2.set_rgba([.75, .35, .9, 1])  # 紫色に変更

    # primitive間の衝突を検出します
    # glass1, star1
    pcd_result1 = glass1.is_pcdwith(star1)
    print("pcd_result(bw glass1 and star1):{}".format(pcd_result1))  # 衝突の結果を出力します
    # star1, star2
    pcd_result2 = star1.is_pcdwith(star2)
    print("pcd_result(bw  star1 and star2):{}".format(pcd_result2))  # 衝突の結果を出力します
    # glass2, star2
    pcd_result3 = glass2.is_pcdwith(star2)
    print("pcd_result(bw glass2 and star2):{}".format(pcd_result3))  # 衝突の結果を出力します

    # mesh間の衝突を検出します
    # glass1, star1
    mcd_result1, cd_points1 = glass1.is_mcdwith(star1, toggle_contacts=True)
    for pnt in cd_points1:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    print("mcd_result(bw glass1 and star1):{}".format(mcd_result1))  # 衝突の結果を出力します
    # star1, star2
    mcd_result2, cd_points2 = star1.is_mcdwith(star2, toggle_contacts=True)
    for pnt in cd_points2:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    print("mcd_result(bw  star1 and star2):{}".format(mcd_result2))  # 衝突の結果を出力します
    # glass2, star2
    mcd_result3, cd_points3 = glass2.is_mcdwith(star2, toggle_contacts=True)
    for pnt in cd_points3:
        gm.gen_sphere(pos=pnt, rgba=[1, 0, 0, 1], radius=.002).attach_to(base)
    print("mcd_result(bw glass2 and star2):{}".format(mcd_result3))  # 衝突の結果を出力します

    # 各CollisionModelとそのmeshを表示します
    glass1.attach_to(base)
    glass1.show_cdmesh()
    glass2.attach_to(base)
    glass2.show_cdmesh()
    star1.attach_to(base)
    star1.show_cdmesh()
    star2.attach_to(base)
    star2.show_cdmesh()

    base.run()