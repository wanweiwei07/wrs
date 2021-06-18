import os
import numpy as np
import basis
import math
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.dynamics.bullet.bdmodel as bdm

# base = wd.World(cam_pos=[1000, 300, 1000], lookat_pos=[0, 0, 0], toggle_debug=True)
base = wd.World(cam_pos=[.3, .3, 1], lookat_pos=[0, 0, 0], toggle_debug=False)
base.setFrameRateMeter(True)
objpath = os.path.join(basis.__path__[0], "objects", "bunnysim.stl")
# objpath = os.path.join(basis.__path__[0], "objects", "block.stl")
bunnycm = bdm.BDModel(objpath, mass=1, type="box")

objpath2 = os.path.join(basis.__path__[0], "objects", "bowlblock.stl")
bunnycm2 = bdm.BDModel(objpath2, mass=0, type="triangles", dynamic=False)
bunnycm2.set_rgba(np.array([0, 0.7, 0.7, 1.0]))
bunnycm2.set_pos(np.array([0, 0, 0]))
bunnycm2.start_physics()
base.attach_internal_update_obj(bunnycm2)


def update(bunnycm, task):
    if base.inputmgr.keymap['space'] is True:
        for i in range(1):
            bunnycm1 = bunnycm.copy()
            bunnycm1.set_mass(.1)
            rndcolor = np.random.rand(4)
            rndcolor[-1] = 1
            bunnycm1.set_rgba(rndcolor)
            rotmat = rm.rotmat_from_euler(0, 0, math.pi / 12)
            z = math.floor(i / 100)
            y = math.floor((i - z * 100) / 10)
            x = i - z * 100 - y * 10
            print(x, y, z, "\n")
            bunnycm1.set_homomat(
                rm.homomat_from_posrot(np.array([x * 0.015 - 0.07, y * 0.015 - 0.07, 0.35 + z * 0.015]), rotmat))
            base.attach_internal_update_obj(bunnycm1)
            bunnycm1.start_physics()
    base.inputmgr.keymap['space'] = False
    return task.cont


gm.gen_frame().attach_to(base)
taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

base.run()