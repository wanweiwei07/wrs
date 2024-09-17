import numpy as np
import basis.trimesh_factory as trf
import modeling.geometric_model as mgm
import visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.3, .3, .3]), lookat_pos=np.zeros(3))
    tg_sphere = trf.gen_sphere(pos=np.zeros(3), radius=.05)  # Trimesh型の球を定義
    gm_sphere = mgm.GeometricModel(tg_sphere)  # GeometricModel型のモデルに変換
    gm_sphere.attach_to(base)  # 画面に表示
    base.run()
