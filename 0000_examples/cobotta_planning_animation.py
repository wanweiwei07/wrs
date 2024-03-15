if __name__ == '__main__':
    import math
    import time
    import numpy as np
    import basis.robot_math as rm
    import robot_sim.robots.cobotta.cobotta as cbt
    import motion.probabilistic.rrt_connect as rrtc
    import motion.probabilistic.rrt_star_connect as rrtsc
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm
    import basis.constant as bc

    base = wd.World(cam_pos=[1.5, 1.5, .75], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)

    robot = cbt.Cobotta(enable_cc=True)

    rrtc_planner = rrtc.RRTConnect(robot)


    class Data(object):
        def __init__(self):
            self.robot_attached_list = []
            self.counter = 0
            self.path = []


    animation_data = Data()


    def update(robot, rrtsc_planner, animation_data, task):
        if animation_data.counter >= len(animation_data.path):
            if len(animation_data.robot_attached_list) != 0:
                for robot_attached in animation_data.robot_attached_list:
                    robot_attached.detach()
            animation_data.robot_attached_list.clear()
            animation_data.path = []
            animation_data.counter = 0
        if animation_data.counter == 0:
            while True:
                start_conf = robot.get_jnt_values()
                robot.goto_given_conf(jnt_values=start_conf)
                start_robot_meshmodel = robot.gen_meshmodel(rgb=rm.bc.tab20_list[6], alpha=.7)
                goal_conf = robot.rand_conf()
                robot.goto_given_conf(jnt_values=goal_conf)
                goal_robot_meshmodel = robot.gen_meshmodel(rgb=rm.bc.tab20_list[0], alpha=.7)
                tic = time.time()
                path = rrtsc_planner.plan(start_conf=start_conf,
                                          goal_conf=goal_conf,
                                          ext_dist=.1,
                                          max_time=300,
                                          smoothing_n_iter=100)
                toc = time.time()
                print(toc - tic)
                if path is not None:
                    # print(anime_data.path)
                    animation_data.path = path
                    # input()
                    start_robot_meshmodel.attach_to(base)
                    goal_robot_meshmodel.attach_to(base)
                    animation_data.robot_attached_list.append(start_robot_meshmodel)
                    animation_data.robot_attached_list.append(goal_robot_meshmodel)
                    break
                else:
                    continue
        # if len(anime_data.robot_attached_list) > 2:
        #     for robot_attached in anime_data.robot_attached_list[2:]:
        #         robot_attached.detach()
        conf = animation_data.path[animation_data.counter]
        robot.goto_given_conf(jnt_values=conf)
        # robot_meshmodel = robot.gen_meshmodel(rgb=rm.bc.jet_map(anime_data.counter / len(anime_data.path)),
        #                                       alpha=1)
        robot_meshmodel = robot.gen_meshmodel(alpha=.3)
        robot_meshmodel.attach_to(base)
        animation_data.robot_attached_list.append(robot_meshmodel)
        animation_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot, rrtc_planner, animation_data],
                          appendTask=True)
    base.run()
