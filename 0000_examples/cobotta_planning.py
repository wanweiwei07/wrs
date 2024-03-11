if __name__ == '__main__':
    import math
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.cobotta.cobotta as cbt
    import motion.probabilistic.rrt_connect as rrtc
    import motion.probabilistic.rrt_star_connect as rrtsc
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm

    base = wd.World(cam_pos=[1.5, 1.5, .75], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)

    robot = cbt.Cobotta(enable_cc=True)
    start_conf = robot.get_jnt_values()
    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=rm.bc.winter_map(0.0), alpha=1).attach_to(base)

    goal_conf = robot.rand_conf()
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=rm.bc.winter_map(1.0), alpha=1).attach_to(base)
    rrtc_planner = rrtc.RRTConnect(robot)
    path = rrtc_planner.plan(start_conf=start_conf,
                             goal_conf=goal_conf,
                             ext_dist=.1,
                             max_time=300)
    # rrtsc_planner = rrtsc.RRTStarConnect(robot)
    # path = rrtsc_planner.plan(start_conf=start_conf,
    #                           goal_conf=goal_conf,
    #                           ext_dist=.1,
    #                           max_time=300,
    #                           smoothing_n_iter=100)
    if path is not None:
        n_step = len(path)
        for i, conf in enumerate(path):
            robot.goto_given_conf(conf)
            robot_meshmodel = robot.gen_meshmodel(rgb=rm.bc.winter_map(i/n_step), alpha=.3)
            robot_meshmodel.attach_to(base)
    else:
        print("No available motion found.")
    base.run()
