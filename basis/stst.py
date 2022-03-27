import numpy as np
import matplotlib.pyplot as plt
import visualization.panda.world as world
import robot_math as rm
import modeling.geometric_model as gm

sp2d = rm.gen_2d_spiral_points(max_radius=.2, radial_granularity=.001, tangential_granularity=.01)
plt.plot(sp2d[:,0], sp2d[:,1])
plt.show()

base = world.World(cam_pos=np.array([1, 1, 1]), lookat_pos=np.array([0, 0, 0.25]))
sp = rm.gen_3d_spiral_points(pos=np.array([0, 0, .25]),
                             rotmat=rm.rotmat_from_axangle(np.array([1, 0, 0]), np.pi / 6),
                             max_radius=.20,
                             radial_granularity=.001,
                             tangential_granularity=.01,)
for id in range(len(sp) - 1):
    pnt0 = sp[id, :]
    pnt1 = sp[id + 1, :]
    gm.gen_stick(spos=pnt0, epos=pnt1, type="round").attach_to(base)
base.run()
