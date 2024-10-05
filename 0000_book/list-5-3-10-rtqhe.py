from wrs import wd, rm, rtqhe, mgm

if __name__ == "__main__":
    base = wd.World(cam_pos=rm.np.array([1, 1, 1]), lookat_pos=rm.np.array([0, 0, 0]))
    mgm.gen_frame().attach_to(base)
    gripper = rtqhe.RobotiqHE()
    gripper.change_jaw_width(.05)
    gripper.gen_meshmodel(rgb=rm.np.array([.3, .3, .3]), alpha=.3).attach_to(base)
    gripper.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    gripper.fix_to(pos=rm.np.array([-.07, .14, 0]), rotmat=rm.rotmat_from_axangle(rm.const.x_ax, .05))
    gripper.gen_meshmodel(toggle_cdmesh=True).attach_to(base)
    gripper.fix_to(pos=rm.np.array([.07, -.14, 0]), rotmat=rm.rotmat_from_axangle(rm.const.x_ax, .05))
    gripper.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    base.run()
