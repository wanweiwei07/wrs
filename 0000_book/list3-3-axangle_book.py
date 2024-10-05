from wrs import wd, rm, mgm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
# 座標系O
rotmat_o = rm.np.eye(3)
mgm.gen_frame(rotmat=rotmat_o, ax_length=.2).attach_to(base)
# 座標系A
## オイラーの表現で回転行列を求めます
rotmat_a = rm.rotmat_from_euler(rm.pi / 3, -rm.pi / 6, rm.pi / 3)
mgm.gen_dashed_frame(rotmat=rotmat_a, ax_length=.2).attach_to(base)
# 座標系Oのx,y軸を囲む円を表示します
mgm.gen_torus(axis=rotmat_o[:, 2], major_radius=.2, minor_radius=.0015, rgb=rm.np.array([1, 1, 0]),
              n_sec_major=64).attach_to(base)
# 座標系Aのx,yを囲む円を表示します
mgm.gen_dashed_torus(axis=rotmat_a[:, 2], major_radius=.2, minor_radius=.0015, rgb=rm.np.array([1, 1, 0]),
                     len_interval=.007, len_solid=.01, n_sec_major=64).attach_to(base)
# オイラーで求めた回転行列を軸-角度形式に変換する
ax, angle = rm.axangle_between_rotmat(rm.eye(3), rotmat_a)
# 回転軸を書き出します
mgm.gen_arrow(epos=ax * .4, rgb=rm.np.array([0, 0, 0])).attach_to(base)
# 回転の中間状態10個に離散化して画面に表示します
for step_angle in rm.np.linspace(0, angle, 10).tolist():
    rotmat_ = rm.rotmat_from_axangle(ax, step_angle)
    mgm.gen_dashed_frame(ax_length=.2, rotmat=rotmat_).attach_to(base)
    mgm.gen_dashed_torus(axis=rotmat_[:, 2], major_radius=.2, minor_radius=.0015, rgb=rm.np.array([1, 1, 0]),
                         len_interval=.007, len_solid=.01, n_sec_major=64).attach_to(base)
# 大きい包囲球を描きます
mgm.gen_sphere(radius=.2, rgb=rm.const.gray, alpha=.8, ico_level=5).attach_to(base)
base.run()
