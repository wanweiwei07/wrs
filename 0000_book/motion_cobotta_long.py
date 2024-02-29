if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.cobotta.cobotta as cbt
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import motion.trajectory.piecewisepoly_scl as trajp
    import motion.trajectory.piecewisepoly_opt as trajpopt
    import motion.trajectory.piecewisepoly_toppra as trajptop

    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    while True:
        robot_s = cbt.Cobotta()
        # start_conf = robot_s.get_jnt_values(component_name="arm")
        # print("start_radians", start_conf)
        # tgt_pos = np.array([0, -.2, .15])
        # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
        # jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        # rrtc_planner = rrtc.RRTConnect(robot_s)
        # path = rrtc_planner.plan(component_name="arm",
        #                          start_conf=start_conf,
        #                          goal_conf=jnt_values,
        #                          ext_dist=.1,
        #                          max_time=300)
        current_conf = robot_s.get_jnt_values(component_name="arm")
        # go to 1
        start_conf = current_conf
        print("start_radians 1", start_conf)
        tgt_pos = np.array([.0, .2, .35])
        tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi)
        jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        rrtc_planner = rrtc.RRTConnect(robot_s)
        path_goto1 = rrtc_planner.plan(component_name="arm", start_conf=start_conf, goal_conf=jnt_values, ext_dist=.1,
                                       max_time=300)
        # go to 2
        start_conf = path_goto1[-1]
        print("start_radians 2", start_conf)
        tgt_pos = np.array([.25, -.2, .15])
        tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        rrtc_planner = rrtc.RRTConnect(robot_s)
        path_goto2 = rrtc_planner.plan(component_name="arm", start_conf=start_conf, goal_conf=jnt_values, ext_dist=.1,
                                       max_time=300)
        # return
        start_conf = path_goto2[-1]
        print("start_radians return", start_conf)
        goal_conf = current_conf
        rrtc_planner = rrtc.RRTConnect(robot_s)
        path_return = rrtc_planner.plan(component_name="arm", start_conf=start_conf, goal_conf=goal_conf, ext_dist=.1,
                                        max_time=300)
        path = path_goto1 + path_goto2 + path_return

        for pose in path:
            robot_s.fk("arm", pose)
            robot_meshmodel = robot_s.gen_meshmodel()
            robot_meshmodel.attach_to(base)

        tg = trajp.PiecewisePolyScl(method="quintic")
        # tg = trajpopt.PiecewisePolyOpt(method="cubic")
        # tg = trajptop.PiecewisePolyTOPPRA()
        interpolated_confs = tg.interpolate_by_max_spdacc(path, control_frequency=.008, max_vels=[math.pi / 2] * 6,
                                                          max_accs=[math.pi] * 6, toggle_debug_fine=True,
                                                          toggle_debug=True)
    # base.run()
