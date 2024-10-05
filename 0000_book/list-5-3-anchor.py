from wrs import wd, rm, mgm
import wrs.robot_sim._kinematics.jl as rkjl

if __name__ == "__main__":
    base = wd.World(cam_pos=rm.np.array([1, .5, .5]), lookat_pos=rm.np.array([0, 0, .1]))
    anchor = rkjl.Anchor(n_flange=4, n_lnk=1)
    anchor.loc_flange_pose_list = [[rm.np.array([.1, 0, .2]), rm.rotmat_from_euler(0, 0, 0)],
                                   [rm.np.array([.1, .1, .2]), rm.rotmat_from_euler(-rm.pi / 6, 0, 0)],
                                   [rm.np.array([.1, -.1, .2]), rm.rotmat_from_euler(rm.pi / 6, 0, 0)],
                                   [rm.np.array([0, 0, .2]), rm.rotmat_from_euler(0, -rm.pi / 6, 0)]]
    mgm.gen_frame().attach_to(base)
    anchor.gen_stickmodel().attach_to(base)

    base.run()
