from wrs import wd, rm, ur3d, mgm, mcm

if __name__ == '__main__':
    base = wd.World(cam_pos=[4, 7, 4], lookat_pos=[.4, 0, 1])
    mgm.gen_frame(ax_length=.35).attach_to(base) # グローバル座標系
    obstacle = mcm.CollisionModel("./objects/milkcarton.stl") # 衝突検出用モデルの定義
    obstacle.pos = rm.np.array([.55, -.3, 1.3]) # 位置の設定
    obstacle.rotmat = rm.rotmat_from_euler(-rm.pi/3, rm.pi/6, rm.pi/9) # 回転の設定
    obstacle.rgba = rm.np.array([.5, .7, .3, .5]) # 色の設定
    obstacle.attach_to(base) # 画面への表示
    mgm.gen_frame(ax_length=.15).attach_to(obstacle) # 障害物のローカル座標系
    # ロボットシミュレーション関連
    robot_s = ur3d.UR3Dual() # シミュレーション用のロボットの定義
    robot_model = robot_s.gen_meshmodel(alpha=.5) # 現在の姿勢を用いてメッシュを生成
    robot_model.attach_to(base) # 生成したメッシュを画面に表示
    base.run() # 仮想環境を実行