from wrs import wd, rm, mgm, mcm, rrtc, cbt

base = wd.World(cam_pos=rm.vec(2, 2, .5), lookat_pos=rm.vec(0, 0, .2))
mgm.gen_frame().attach_to(base)
# robot
robot = cbt.Cobotta(enable_cc=True)
# object holder
obj_cmodel = mcm.CollisionModel("objects/holder.stl")
obj_cmodel.rgba = rm.np.array([.5, .5, .5, 1])
obj_cmodel.pos = rm.np.array([.2, 0, .3])
obj_cmodel.rotmat = rm.np.eye(3)
mgm.gen_frame().attach_to(obj_cmodel)
obj_cmodel.attach_to(base)

obj_cmodel2 = obj_cmodel.copy()
obj_cmodel2.pos = rm.np.array([.2, .1, .2])
obj_cmodel2.attach_to(base)
robot.hold(obj_cmodel2)

# robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
print(robot.is_collided([obj_cmodel]))
# base.run()

rrtc_planner = rrtc.RRTConnect(robot)

class AnimeData(object):
    def __init__(self):
        self.robot_attached_list = []
        self.counter = 0
        self.path = []
        self.robot = robot
        self.planner = rrtc_planner
        self.obstacle_list = [obj_cmodel]

anime_data = AnimeData()

def update(animation_data, task):
    if animation_data.counter >= len(animation_data.path):
        if len(animation_data.robot_attached_list) != 0:
            for robot_attached in animation_data.robot_attached_list:
                robot_attached.detach()
        animation_data.robot_attached_list.clear()
        animation_data.path = []
        animation_data.counter = 0
    if animation_data.counter == 0:
        while True:
            start_conf = animation_data.robot.get_jnt_values()
            animation_data.robot.goto_given_conf(jnt_values=start_conf)
            start_robot_meshmodel = animation_data.robot.gen_meshmodel(rgb=rm.const.tab20_list[6], alpha=.7)
            goal_conf = animation_data.robot.rand_conf()
            animation_data.robot.goto_given_conf(jnt_values=goal_conf)
            goal_robot_meshmodel = animation_data.robot.gen_meshmodel(rgb=rm.const.tab20_list[0], alpha=.7)
            path = animation_data.planner.plan(start_conf=start_conf,
                                               goal_conf=goal_conf,
                                               ext_dist=.1,
                                               obstacle_list=animation_data.obstacle_list,
                                               max_time=300,
                                               smoothing_n_iter=30)
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
    conf = animation_data.path[animation_data.counter][0]
    animation_data.robot.goto_given_conf(jnt_values=conf)
    # robot_meshmodel = anime_data.robot.gen_meshmodel(rgb=rm.const.jet_map(anime_data.counter / len(anime_data.path)),
    #                                       alpha=1)
    robot_meshmodel = animation_data.robot.gen_meshmodel(toggle_cdprim=False, alpha=.7)
    robot_meshmodel.attach_to(base)
    animation_data.robot_attached_list.append(robot_meshmodel)
    animation_data.counter += 1
    print(animation_data.counter)
    # if anime_data.counter > 30 and len(anime_data.robot.end_effector.oiee_list)>0:
    #     anime_data.robot.release(obj_cmodel2)
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)
base.run()
