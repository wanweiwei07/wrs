import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtq_he
import modeling.geometric_model as gm

if __name__ == "__main__":
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame(length=.2).attach_to(base)
    grpr = rtq_he.RobotiqHE(enable_cc=True)
    grpr.jaw_to(.05)
    grpr.gen_meshmodel(rgba=[.3,.3,.3,.3]).attach_to(base)
    grpr.gen_stickmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)
    grpr.fix_to(pos=np.array([-.1, .2, 0]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdmesh()
    grpr.fix_to(pos=np.array([.1, -.2, 0]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    grpr.show_cdprimit()
    base.run()