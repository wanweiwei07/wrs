import numpy as np
from wrs import basis as rm, modeling as gm
import wrs.visualization.panda.world as wd

leaf_rgba = [45/255, 90/255, 39/255, 1]
stem_rgba = [97/255, 138/255, 61/255, 1]

base = wd.World(cam_pos=[1,1,.7])
sb_leaf = gm.GeometricModel(initor="objects/soybean_leaf.stl")
sb_leaf.set_rgba(rgba=leaf_rgba)

stem0_spos = np.array([0,0,0])
stem0_epos = np.array([0,0,.05])
stem0 = gm.gen_stick(spos=stem0_spos, epos=stem0_epos, rgba=stem_rgba)
stem0.attach_to(base)

sbl0 = sb_leaf.copy()
sbl0.set_pos(stem0_epos)
sbl0.attach_to(base)

sbl1 = sb_leaf.copy()
sbl1.set_pos(stem0_epos-np.array([0,0,0.005]))
sbl1.set_rotmat(rm.rotmat_from_axangle([0,0,1], np.pi))
sbl1.attach_to(base)

base.run()