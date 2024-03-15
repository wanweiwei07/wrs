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
            self.counter = 0
            self.mdata = None


    anime_data = Data()


    def update(robot, rrtsc_planner, anime_data, task):
        if anime_data.mdata is not None and anime_data.counter >= len(anime_data.mdata):
            for mesh_model in anime_data.mdata.mesh_list:
                mesh_model.detach()
            anime_data.counter = 0
        if anime_data.counter == 0:
            while True:
                start_conf = robot.get_jnt_values()
                goal_conf = robot.rand_conf()
                tic = time.time()
                mdata = rrtsc_planner.plan(start_conf=start_conf,
                                           goal_conf=goal_conf,
                                           ext_dist=.1,
                                           max_time=300,
                                           smoothing_n_iter=100)
                toc = time.time()
                print(toc - tic)
                if mdata is not None:
                    mdata.mesh_list[0].rgb = rm.bc.tab20_list[7]
                    mdata.mesh_list[0].attach_to(base)
                    mdata.mesh_list[-1].rgb = rm.bc.tab20_list[0]
                    mdata.mesh_list[-1].attach_to(base)
                    anime_data.mdata = mdata
                    anime_data.counter += 1
                    break
                else:
                    continue
        mesh_model = anime_data.mdata.mesh_list[anime_data.counter]
        mesh_model.alpha = .3
        mesh_model.attach_to(base)
        anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot, rrtc_planner, anime_data],
                          appendTask=True)
    base.run()
