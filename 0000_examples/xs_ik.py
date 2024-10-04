import time
import math
import numpy as np
from wrs.basis import robot_math as rm
import wrs.visualization.panda.world as wd
from wrs import robot_sim as xss, modeling as gm, modeling as cm

base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("./objects/bunnysim.stl")
object.set_pos(np.array([.85, 0, .37]))
object.set_rgba([.5, .7, .3, 1])
object.attach_to(base)
# robot_s
component_name = 'arm'
robot_s = xss.XArmShuidi()
robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
# base.run()
seed_jnt_values = robot_s.get_jnt_values(component_name=component_name)
for y in range(-5, 5):
    tgt_pos = np.array([.3, y*.1, 1])
    tgt_rotmat = rm.rotmat_from_euler(0, math.pi / 2, 0)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = robot_s.ik(component_name=component_name,
                            tgt_pos=tgt_pos,
                            tgt_rotmat=tgt_rotmat,
                            max_niter=500,
                            toggle_debug=False,
                            seed_jnt_values=seed_jnt_values)
    toc = time.time()
    print(f"time cost: {toc-tic}")
    seed_jnt_values = jnt_values
    if jnt_values is not None:
        robot_s.fk(component_name=component_name, jnt_values=jnt_values)
        robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
base.run()