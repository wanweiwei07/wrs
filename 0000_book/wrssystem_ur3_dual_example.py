import wrs.visualization.panda.world as wd
import math
import numpy as np
from wrs import basis as rm, robot_sim as ur3d, modeling as gm, modeling as cm

if __name__ == '__main__':
    base = wd.World(cam_pos=[9, 4, 4], lookat_pos=[0, 0, .7])
    gm.gen_frame(axis_length=.7, axis_radius=.02).attach_to(base)
    # object
    object = cm.CollisionModel("./objects/bunnysim.stl")
    object.set_pos(np.array([.55, -.3, 1.3]))
    object.set_rotmat(rm.rotmat_from_euler(-math.pi/3, math.pi/6, math.pi/9))
    object.set_rgba([.5, .7, .3, 1])
    object.attach_to(base)
    gm.gen_frame(axis_length=.3, axis_radius=.015).attach_to(object)
    # robot_s
    robot_s = ur3d.UR3Dual()
    robot_meshmodel = robot_s.gen_meshmodel(rgba=[.3,.3,.3,.3])
    robot_meshmodel.attach_to(base)
    base.run()