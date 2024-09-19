import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import robot_sim._kinematics.jl as rkjl
import robot_sim._kinematics.jlchain as rkjlc
import modeling.geometric_model as mgm

if __name__ == "__main__":
    base = wd.World(cam_pos=[1, .5, .5], lookat_pos=[0, 0, .1])
    anchor = rkjl.Anchor(n_flange=4, n_lnk=1)
    anchor.loc_flange_pose_list = [[np.array([.1, 0, .2]), rm.rotmat_from_euler(0, 0, 0)],
                                   [np.array([.1, .1, .2]), rm.rotmat_from_euler(-np.pi / 6, 0, 0)],
                                   [np.array([.1, -.1, .2]), rm.rotmat_from_euler(np.pi / 6, 0, 0)],
                                   [np.array([0, 0, .2]), rm.rotmat_from_euler(0, -np.pi / 6, 0)]]
    mgm.gen_frame().attach_to(base)
    anchor.gen_stickmodel().attach_to(base)


    base.run()
