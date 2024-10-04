import numpy as np
from wrs import basis as tg, modeling as gm
import wrs.visualization.panda.world as wd

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.3,.3,.3]), lookat_pos=np.zeros(3))
    tg_sphere = tg.gen_sphere(pos=np.zeros(3), radius=.05)
    gm_sphere = gm.GeometricModel(tg_sphere)
    gm_sphere.attach_to(base)
    base.run()
