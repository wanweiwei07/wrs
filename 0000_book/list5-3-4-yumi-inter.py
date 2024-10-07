from wrs import wd, rm, ym, mip

if __name__ == '__main__':
    base = wd.World(cam_pos=rm.vec(3, 1, 1), lookat_pos=rm.vec(0, 0, 0.5))
    robot = ym.Yumi(enable_cc=True)
    robot.use_rgt()
    interp_planner = mip.InterplatedMotion(robot=robot)
    start_pos = rm.np.array([.6, -.3, .5])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], rm.pi / 2)
    goal_pos = rm.np.array([.6, .0, .3])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], rm.pi)
    mot_data = interp_planner.gen_linear_motion(start_tcp_pos=start_pos, start_tcp_rotmat=start_rotmat,
                                                goal_tcp_pos=goal_pos, goal_tcp_rotmat=goal_rotmat,
                                                toggle_dbg=True)
    for mesh in mot_data.mesh_list:
        mesh.attach_to(base)
    base.run()
