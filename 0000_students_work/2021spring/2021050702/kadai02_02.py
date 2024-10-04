# kadai02_02.py
# 2021/04/30 実験C第二回の課題2
# Yumiロボットの作業空間において，2つの障害物を回避する動作の計画を実行しました．

import numpy as np
import math
import time
from wrs import basis as rm, robot_sim as ym, motion as rrtc, modeling as gm, modeling as cm
import wrs.visualization.panda.world as wd

if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0.2, 1.2], lookat_pos=[0, 0, 0])
    gm.gen_frame(axis_length=.2).attach_to(base)  # グローバル座標系
    yumi_s = ym.Yumi(enable_cc=True)  # Yumiインスタンスを定義します
    component_name = 'rgt_arm'
    rrtc_planner = rrtc.RRTConnect(yumi_s)  # 動作計画器

    # 初期姿勢における関節変位を求めます
    start_pos = np.array([.15, -.35, .3])  # 初期姿勢
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi/2)  # 初期姿勢
    gm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    start_conf = yumi_s.ik(component_name, start_pos, start_rotmat)
    # ゴールの姿勢における関節変位を求めます
    goal_pos = np.array([.4, -.2, .2])  # ゴールの姿勢
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi/3)  # ゴールの姿勢
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    goal_conf = yumi_s.ik(component_name, goal_pos, goal_rotmat)

    # 障害物を配置します
    glass1 = cm.CollisionModel(initor="objects/glass1.stl", cdprim_type="surface_balls", cdmesh_type="convex_hull")
    glass1.set_rgba([.9, .75, .35, 1])  # 黄色に変更
    glass1.set_pos(np.array([.25, -.3, .2]))
    star1 = cm.CollisionModel(initor="objects/star2.stl", cdprim_type="surface_balls", cdmesh_type="convex_hull")
    star1.set_rgba([.9, .75, .35, 1])  # 黄色に変更
    star1.set_pos(np.array([.25, -0.05, .15]))
    star1.set_rotmat(rm.rotmat_from_axangle(axis=[0, 0, 1], angle=math.pi / 2.))
    # 各障害物のCollisionModelを表示します
    glass1.attach_to(base)
    #glass1.show_cdprimit()
    star1.attach_to(base)
    #star1.show_cdprimit()
    # obstacle listを作成します
    obstacle_list = [glass1, star1]

    # 動作計画をします
    print("planning ... ", end="")
    tic = time.time()
    path = rrtc_planner.plan(component_name=component_name,
                             start_conf=start_conf,
                             goal_conf=goal_conf,
                             obstacle_list=obstacle_list,
                             ext_dist=.2,
                             max_time=300)
    toc = time.time()
    print("done. ({} sec)".format(toc - tic))

    # スタートから順番に画面に表示します
    yumi_s.fk(component_name, start_conf)
    yumi_meshmodel = yumi_s.gen_meshmodel(toggle_tcpcs=False)
    yumi_meshmodel.attach_to(base)
    yumi_s.gen_stickmodel().attach_to(base)
    for pose in path:
        yumi_s.fk(component_name, pose)
        yumi_meshmodel = yumi_s.gen_meshmodel(toggle_tcpcs=False)
        yumi_meshmodel.attach_to(base)
        yumi_s.gen_stickmodel().attach_to(base)
    base.run()
