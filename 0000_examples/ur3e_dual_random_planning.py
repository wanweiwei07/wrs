import random
import numpy as np
import wrs.visualization.panda.world as wd
import wrs.robot_sim.robots.ur3e_dual.ur3e_dual as u3ed
from wrs import motion as rrtc, modeling as gm, modeling as cm


class Data(object):
    def __init__(self):
        self.counter = 0
        self.mot_data = None


base = wd.World(cam_pos=[3, 2, 3], lookat_pos=[0.5, 0, 1.1])
gm.gen_frame().attach_to(base)
# object
object = cm.CollisionModel("objects/bunnysim.stl")
object.pos = np.array([.75, .2, 1.3])
object.rgba = np.array([.5, .7, .3, 1])
object.attach_to(base)
# robot
robot = u3ed.UR3e_Dual()
# robot.use_lft()
# planner
rrtc_planner = rrtc.RRTConnect(robot)

anime_data = Data()


def update(robot, rrtc_planner, anime_data, obstacle_list, task):
    if anime_data.counter == 0:
        value = random.choice([1, 2, 3])
        if value == 1:
            robot.use_both()
        elif value == 2:
            robot.use_lft()
        else:
            robot.use_rgt()
        while True:
            # plan
            start_conf = robot.get_jnt_values()
            goal_conf = robot.rand_conf()
            mot_data = rrtc_planner.plan(start_conf=start_conf,
                                         goal_conf=goal_conf,
                                         obstacle_list=obstacle_list,
                                         ext_dist=.1,
                                         max_time=30,
                                         smoothing_n_iter=100)
            if mot_data is not None:
                robot.goto_given_conf(jnt_values=goal_conf)
                # print(anime_data.path)
                anime_data.mot_data = mot_data
                anime_data.mot_data.mesh_list[anime_data.counter].attach_to(base)
                anime_data.counter = 1
                break
            else:
                continue
    if base.inputmgr.keymap['space']:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            anime_data.mot_data = None
            anime_data.counter = 0
        else:
            anime_data.mot_data.mesh_list[anime_data.counter].attach_to(base)
            anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot, rrtc_planner, anime_data, [object]],
                      appendTask=True)
base.run()
