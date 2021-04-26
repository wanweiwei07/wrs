import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.ur3_dual.ur3_dual as ur3d
import numpy as np

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
    gm.gen_frame().attach_to(base)
    # object
    object = cm.CollisionModel("./objects/bunnysim.stl")
    object.set_pos(np.array([.55, -.3, 1.3]))
    object.set_rgba([.5, .7, .3, 1])
    object.attach_to(base)
    # robot_s
    robot_s = ur3d.UR3Dual()
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    base.run()