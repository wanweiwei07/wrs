import time
from wrs import basis as rm, robot_sim as shuidi, motion as rrtc, modeling as mgm
import wrs.visualization.panda.world as wd

if __name__ == "__main__":

    base = wd.World(cam_pos=[1.5, 1.5, .75], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)

    robot = shuidi.Shuidi(enable_cc=True)
    rrtc_planner = rrtc.RRTConnect(robot)


    class Data(object):
        def __init__(self):
            self.counter = 0
            self.mot_data = None


    anime_data = Data()


    def update(robot, rrtsc_planner, anime_data, task):
        if anime_data.mot_data is not None and anime_data.counter >= len(anime_data.mot_data):
            for mesh_model in anime_data.mot_data.mesh_list:
                mesh_model.detach()
            anime_data.counter = 0
        if anime_data.counter == 0:
            while True:
                start_conf = robot.get_jnt_values()
                goal_conf = robot.rand_conf()
                tic = time.time()
                mot_data = rrtsc_planner.plan(start_conf=start_conf,
                                              goal_conf=goal_conf,
                                              ext_dist=.1,
                                              max_time=300,
                                              smoothing_n_iter=100)
                toc = time.time()
                print(toc - tic)
                if mot_data is not None:
                    print(mot_data)
                    mot_data.mesh_list[0].rgb = rm.bc.tab20_list[7]
                    mot_data.mesh_list[0].attach_to(base)
                    mot_data.mesh_list[-1].rgb = rm.bc.tab20_list[0]
                    mot_data.mesh_list[-1].attach_to(base)
                    anime_data.mot_data = mot_data
                    anime_data.counter += 1
                    break
                else:
                    continue
        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.alpha = .3
        mesh_model.attach_to(base)
        anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot, rrtc_planner, anime_data],
                          appendTask=True)
    base.run()