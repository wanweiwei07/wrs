if __name__ == '__main__':
    from wrs import wd, rm, mgm, cbt, rrtc, toppra

    base = wd.World(cam_pos=rm.np.array([1, 1, .5]), lookat_pos = rm.np.array([0, 0, .2]))
    mgm.gen_frame().attach_to(base)

    while True:
        robot = cbt.Cobotta()
        start_jnt_values = robot.get_jnt_values()
        print("start_radians", start_jnt_values)
        tgt_pos = rm.np.array([0, -.2, .15])
        tgt_rotmat = rm.rotmat_from_axangle(rm.const.y_ax, rm.pi / 3)
        goal_jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        planner = rrtc.RRTConnect(robot)
        mot_data = planner.plan(start_conf=start_jnt_values,
                            goal_conf=goal_jnt_values,
                            ext_dist=.1,
                            max_time=300)
        # jnt_values = robot.get_jnt_values()
        # # go to 1
        # start_conf = current_conf
        # print("start_radians 1", start_conf)
        # tgt_pos = rm.np.array([.0, .2, .35])
        # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi)
        # jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        # rrtc_planner = rrtc.RRTConnect(robot_s)
        # path_goto1 = rrtc_planner.plan(component_name="arm", start_conf=start_conf, goal_conf=jnt_values, ext_dist=.1,
        #                                max_time=300)
        # # go to 2
        # start_conf = path_goto1[-1]
        # print("start_radians 2", start_conf)
        # tgt_pos = rm.np.array([.25, -.2, .15])
        # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        # jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        # rrtc_planner = rrtc.RRTConnect(robot_s)
        # path_goto2 = rrtc_planner.plan(component_name="arm", start_conf=start_conf, goal_conf=jnt_values, ext_dist=.1,
        #                                max_time=300)
        # # return
        # start_conf = path_goto2[-1]
        # print("start_radians return", start_conf)
        # goal_conf = current_conf
        # rrtc_planner = rrtc.RRTConnect(robot_s)
        # path_return = rrtc_planner.plan(component_name="arm", start_conf=start_conf, goal_conf=goal_conf, ext_dist=.1,
        #                                 max_time=300)
        # path = path_goto1 + path_goto2 + path_return

        for conf in mot_data:
            robot.fk(conf)
            robot.gen_meshmodel().attach_to(base)

        interp_time, interp_confs, interp_spds, interp_accs = toppra.generate_time_optimal_trajectory(
            mot_data.jv_list, max_vels=[rm.pi / 2] * 6, max_accs=[rm.pi] * 6, ctrl_freq=.008)
        break

    base.run()
