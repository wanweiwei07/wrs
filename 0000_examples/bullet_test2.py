import os
from wrs import wd, rm, mgm
import wrs.modeling.dynamics.bullet.bdmodel as bdm

# base = wd.World(cam_pos=[1000, 300, 1000], lookat_pos=[0, 0, 0], toggle_dbg=True)
base = wd.World(cam_pos=[.3, .3, 1], lookat_pos=[0, 0, 0], toggle_debug=False)
base.setFrameRateMeter(True)
obj_path = os.path.join(os.path.dirname(rm.__file__), "objects", "bunnysim.stl")
# obj_path = os.path.join(basis.__path__[0], "objects", "block.stl")
bunnycm = bdm.BDModel(obj_path, mass=1, type="box")

bunnycm2 = bdm.BDModel(obj_path, mass=0, type="triangles", dynamic=False)
bunnycm2.set_rgba(rm.np.array([0, 0.7, 0.7, 1.0]))
bunnycm2.set_pos(rm.np.array([0, 0, 0]))
bunnycm2.start_physics()
base.attach_internal_update_obj(bunnycm2)


def update(bunnycm, task):
    if base.inputmgr.keymap['space'] is True:
        for i in range(1):
            bunnycm1 = bunnycm.copy()
            bunnycm1.set_mass(.1)
            rndcolor = rm.np.random.rand(4)
            rndcolor[-1] = 1
            bunnycm1.set_rgba(rndcolor)
            rotmat = rm.rotmat_from_euler(0, 0, rm.pi / 12)
            z = rm.floor(i / 100)
            y = rm.floor((i - z * 100) / 10)
            x = i - z * 100 - y * 10
            print(x, y, z, "\n")
            bunnycm1.set_homomat(
                rm.homomat_from_posrot(rm.np.array([x * 0.015 - 0.07, y * 0.015 - 0.07, 0.35 + z * 0.015]), rotmat))
            base.attach_internal_update_obj(bunnycm1)
            bunnycm1.start_physics()
    base.inputmgr.keymap['space'] = False
    return task.cont


mgm.gen_frame().attach_to(base)
taskMgr.add(update, "addobject", extraArgs=[bunnycm], appendTask=True)

base.run()