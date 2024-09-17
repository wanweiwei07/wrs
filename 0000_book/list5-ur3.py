import visualization.panda.world as wd # 三次元の仮想環境や表示画面の定義用
import modeling.geometric_model as mgm # 各種幾何形状（例えば矢印や座標系など）の定義用
import modeling.collision_model as mcm # 衝突検出可能な各種幾何形状の定義用
import robot_sim.robots.ur3_dual.ur3_dual as ur3d # ロボットシミュレーションの定義用
import numpy as np # 行列計算用
import basis.robot_math as rm # 座標計算用

if __name__ == '__main__':
    base = wd.World(cam_pos=[7, 2, 4], lookat_pos=[0, 0, 1])
    mgm.gen_frame(ax_length=.7, ax_radius=.01).attach_to(base) # グローバル座標系
    obstacle = mcm.CollisionModel("./objects/milkcarton.stl") # 衝突検出用モデルの定義
    obstacle.pos = np.array([.55, -.3, 1.3]) # 位置の設定
    obstacle.rotmat = rm.rotmat_from_euler(-np.pi/3, np.pi/6, np.pi/9) # 回転の設定
    obstacle.rgba = np.array([.5, .7, .3, .5]) # 色の設定
    obstacle.attach_to(base) # 画面への表示
    mgm.gen_frame(ax_length=.3, ax_radius=.007).attach_to(obstacle) # 障害物のローカル座標系
    # ロボットシミュレーション関連
    robot_s = ur3d.UR3Dual() # シミュレーション用のロボットの定義
    robot_model = robot_s.gen_meshmodel(alpha=.5) # 現在の姿勢を用いてメッシュを生成
    robot_model.attach_to(base) # 生成したメッシュを画面に表示
    base.run() # 仮想環境を実行