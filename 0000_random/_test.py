from wrs import wd, mgm
import numpy as np

base = wd.World(cam_pos=np.array([.5,  0,1]), auto_rotate=True)
for i in range(2000):
    pos = np.random.rand(3)
    print(pos)
    mgm.gen_frame(pos=pos).attach_to(base)
base.run()