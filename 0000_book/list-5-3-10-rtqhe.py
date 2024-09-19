import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtq_he
import modeling.geometric_model as mgm

if __name__ == "__main__":
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    grpr = rtq_he.RobotiqHE()
    grpr.change_jaw_width(.05)
    grpr.gen_meshmodel(rgb=np.array([.3,.3,.3]), alpha=.3).attach_to(base)
    grpr.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    grpr.fix_to(pos=np.array([-.07, .14, 0]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel(toggle_cdmesh=True).attach_to(base)
    grpr.fix_to(pos=np.array([.07, -.14, 0]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    base.run()