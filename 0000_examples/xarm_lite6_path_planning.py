import numpy as np

if __name__ == "__main__":
    import modeling.geometric_model as gm
    import visualization.panda.world as wd
    from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper
    import motion.probabilistic.rrt_connect as rrtc
    import math

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    # initialize
    xarm = XArmLite6WRSGripper(enable_cc=True)
    component_name = 'arm'

    start_conf = np.array([-1.447399, -0.139943, 0.376222, -1.302925, 1.107183, 0.524813])
    goal_conf = np.array([0.796652, -0.139604, 0.914931, 1.328044, 0.434468, 0.989801])

    xarmlite6_cm = xarm.gen_meshmodel()
    xarmlite6_cm.attach_to(base)
    rrtc_planner = rrtc.RRTConnect(xarm)
    path = rrtc_planner.plan(component_name=component_name,
                             start_conf=start_conf,
                             goal_conf=goal_conf,
                             obstacle_list=[],
                             ext_dist=.2,
                             max_time=300)

    # plot the animation
    if path is not None:
        plot_node = [xarmlite6_cm]
        counter = [0]


        def plot_rbt_realtime(task):
            if counter[0] >= len(path):
                counter[0] = 0
            if plot_node[0] is not None:
                plot_node[0].detach()
            pose = path[counter[0]]
            xarm.fk(component_name, pose)
            plot_node[0] = xarm.gen_meshmodel()
            plot_node[0].attach_to(base)
            counter[0] += 1
            return task.again


        base.taskMgr.doMethodLater(0.1, plot_rbt_realtime, "plot robot")

        real_robot = True
        if real_robot:
            from robot_con.xarm_lite6 import XArmLite6X

            xarm_con = XArmLite6X()
            # move to first joint of the path
            suc = xarm_con.move_j(path[0], )
            # # execute RRT motion
            jnt_ground_truth, r = xarm_con.move_jntspace_path(path, )

    base.run()
