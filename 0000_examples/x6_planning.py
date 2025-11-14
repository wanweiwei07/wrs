from wrs import wd, rm, rrtc, mgm, x6wg2


base = wd.World(cam_pos=rm.vec(2, 0, 1), lookat_pos=rm.vec(0, 0, 0.5))
mgm.gen_frame().attach_to(base)
# initialize
robot = x6wg2.XArmLite6WG2(enable_cc=True)

start_conf = rm.np.array([-1.447399, -0.139943, 0.376222, -1.302925, 1.107183, 0.524813])
goal_conf = rm.np.array([0.796652, -0.139604, 0.914931, 1.328044, 0.434468, 0.989801])
start_pos, start_rotmat = robot.fk(start_conf)
goal_pos, goal_rotmat = robot.fk(goal_conf)
mgm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)

planner = rrtc.RRTConnect(robot=robot)
mot_data = planner.plan(start_conf=start_conf,
                    goal_conf=goal_conf,
                    obstacle_list=[],
                    ext_dist=.2,
                    max_time=300)

# plot the animation
if mot_data is not None:
    plot_node = [None]
    counter = [0]

    def plot_rbt_realtime(task):
        if counter[0] >= len(mot_data):
            counter[0] = 0
        if plot_node[0] is not None:
            plot_node[0].detach()
        jnt_values = mot_data[counter[0]][0]
        robot.goto_given_conf(jnt_values=jnt_values)
        plot_node[0] = robot.gen_meshmodel()
        plot_node[0].attach_to(base)
        counter[0] += 1
        return task.again


    base.taskMgr.doMethodLater(0.1, plot_rbt_realtime, "plot robot")

    real_robot = False
    if real_robot:
        import wrs.robot_con.xarm_lite6.xarm_lite6_x as x6x

        xarm_con = x6x.XArmLite6X()
        # move to first joint of the path
        suc = xarm_con.move_j(mot_data[0][0], )
        # # execute RRT motion
        jnt_ground_truth, r = xarm_con.move_jntspace_path(mot_data.jv_list, )

base.run()
