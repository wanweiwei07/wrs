if __name__ == '__main__':
    import math
    import numpy as np
    from wrs import basis as rm, robot_sim as cbt, motion as rrtc, modeling as gm
    import wrs.visualization.panda.world as wd
    import wrs.motion.trajectory.piecewisepoly_toppra as trajptop

    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    while True:
        robot_s = cbt.Cobotta()
        start_conf = robot_s.get_jnt_values(component_name="arm")
        print("start_radians", start_conf)
        tgt_pos = np.array([0, -.2, .15])
        tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 3)
        jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        rrtc_planner = rrtc.RRTConnect(robot_s)
        path = rrtc_planner.plan(component_name="arm",
                                 start_conf=start_conf,
                                 goal_conf=jnt_values,
                                 ext_dist=.1,
                                 max_time=300)

        for pose in path:
            robot_s.fk("arm", pose)
            robot_meshmodel = robot_s.gen_meshmodel()
            robot_meshmodel.attach_to(base)

        # tg = trajp.PiecewisePolyScl(method="quintic")
        # tg = trajpopt.PiecewisePolyOpt(method="quintic")
        # tg = trajpsecopt.PiecewisePolySectionOpt(method="quintic")
        tg = trajptop.PiecewisePolyTOPPRA()
        interpolated_confs = tg.interpolate_by_max_spdacc(path, control_frequency=.008, max_vels=[math.pi / 2] * 6,
                                                          max_accs=[math.pi] * 6, toggle_debug=True)
    # base.run()
