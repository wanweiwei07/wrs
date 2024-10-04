import random
import wrs.motion.probabilistic.rrt_star_connect as rrtsc
import wrs.visualization.panda.world as wd
from wrs import basis as bc, robot_sim as ybt, motion as rrtc, modeling as mgm

base = wd.World(cam_pos=[3, 3, 1.5], lookat_pos=[0, 0, .35])
mgm.gen_frame().attach_to(base)

robot = ybt.Yumi(enable_cc=True)

rrtc_planner = rrtc.RRTConnect(robot)
rrtsc_planner = rrtsc.RRTStarConnect(robot)

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
        value = random.choice([1,2,3])
        if value == 1:
            robot.use_both()
        elif value == 2:
            robot.use_lft()
        else:
            robot.use_rgt()
        while True:
            # plan
            start_conf = robot.get_jnt_values()
            robot.goto_given_conf(jnt_values=start_conf)
            start_robot_meshmodel = robot.gen_meshmodel(rgb=bc.red, alpha=1, toggle_cdprim=True)
            goal_conf = robot.rand_conf()
            robot.goto_given_conf(jnt_values=goal_conf)
            goal_robot_meshmodel = robot.gen_meshmodel(rgb=bc.blue, alpha=1, toggle_cdprim=True)
            path = rrtsc_planner.plan(start_conf=start_conf,
                                                     goal_conf=goal_conf,
                                                     ext_dist=.1,
                                                     max_time=300,
                                                     smoothing_n_iter=100)
            if path is not None:
                # print(anime_data.path)
                animation_data.path=path
                # input()
                start_robot_meshmodel.attach_to(base)
                goal_robot_meshmodel.attach_to(base)
                animation_data.robot_attached_list.append(start_robot_meshmodel)
                animation_data.robot_attached_list.append(goal_robot_meshmodel)
                break
            else:
                continue
    if len(animation_data.robot_attached_list) > 2:
        for robot_attached in animation_data.robot_attached_list[2:]:
            robot_attached.detach()
    conf = animation_data.path[animation_data.counter]
    robot.goto_given_conf(jnt_values=conf)
    robot_meshmodel = robot.gen_meshmodel(alpha=1, toggle_cdprim=True)
    robot_meshmodel.attach_to(base)
    animation_data.robot_attached_list.append(robot_meshmodel)
    animation_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot, rrtc_planner, animation_data],
                      appendTask=True)
base.run()
