from wrs import wd, rm, mgm
import wrs.basis.trimesh_factory as trf

if __name__ == '__main__':
    base = wd.World(cam_pos=rm.np.array([.3, .3, .3]), lookat_pos=rm.np.zeros(3))
    tg_sphere = trf.gen_sphere(pos=rm.np.zeros(3), radius=.05)  # Trimesh型の球を定義
    gm_sphere = mgm.GeometricModel(tg_sphere)  # GeometricModel型のモデルに変換
    gm_sphere.attach_to(base)  # 画面に表示
    base.run()
