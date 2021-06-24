import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd

base = wd.World(cam_pos=[1, .7, .3], lookat_pos=[0, 0, 0])
for x in [-.03, 0, .03]:
    for y in [-.03, 0, .03]:
        for z in [-.03, 0, .03]:
            homomat = np.eye(4)
            homomat[:3,3] = np.array([x,y,z])
            gm.gen_frame_box(extent=[.03, .03, .03], homomat=homomat, rgba=[0, 0, 0, 1], thickness=.00001).attach_to(base)
for x in [0]:
    for y in [0]:
        for z in [-.03, 0, .03]:
            homomat = np.eye(4)
            homomat[:3,3] = np.array([x,y,z])
            gm.gen_box(extent=[.03,.03,.03], homomat=homomat, rgba=[1,1,0,1]).attach_to(base)
homomat = np.eye(4)
homomat[:3,3] = np.array([.03,0,-.03])
gm.gen_box(extent=[.03,.03,.03], homomat=homomat, rgba=[1,1,0,1]).attach_to(base)
base.run()
