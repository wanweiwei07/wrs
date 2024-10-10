import os
from wrs import wd, rm, rrtc, mcm, mgm, gg, ko2fg, ppp

mesh_name = "bracketR1"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name+".stl")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name+".pickle")

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=1)
ground.pos = rm.np.array([0, 0, -.5])
ground.attach_to(base)
# ground.show_cdprim()
## object start
obj = mcm.CollisionModel(mesh_path)
obj.rgba = rm.np.array([.5, .5, .5, 1])
obj_gl_pos = rm.np.array([-.3, -.5, .04])
obj_gl_rotmat = rm.rotmat_from_euler(0, 0, rm.pi)
obj.pos = obj_gl_pos
obj.rotmat = obj_gl_rotmat
mgm.gen_frame().attach_to(obj)
# visualize a copy
obj_start = obj.copy()
obj_start.attach_to(base)
obj_start.show_cdprim()
## object goal
obj_goal = obj.copy()
obj_goal_gl_pos = rm.np.array([.5, -.3, .04])
obj_goal_gl_rotmat = rm.rotmat_from_euler(0, 0, rm.pi / 3)
obj_goal.pos = obj_goal_gl_pos
obj_goal.rotmat = obj_goal_gl_rotmat
obj_goal.rgb = rm.const.tab20_list[0]
obj_goal.alpha = .3
obj_goal.attach_to(base)
obj_goal.show_cdprim()
## robot
robot = ko2fg.KHI_OR2FG7()
robot.gen_meshmodel().attach_to(base)
robot.cc.show_cdprim()
# base.run()

rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

grasp_collection = gg.GraspCollection.load_from_disk(grasp_path)
start_conf = robot.get_jnt_values()

goal_pose_list = [(obj_goal_gl_pos, obj_gl_rotmat)]
mot_data = ppp.gen_pick_and_place(obj_cmodel=obj,
                                  end_jnt_values=start_conf,
                                  grasp_collection=grasp_collection,
                                  goal_pose_list=goal_pose_list,
                                  approach_distance_list=[.05] * len(goal_pose_list),
                                  depart_distance_list=[.05] * len(goal_pose_list),
                                  pick_approach_distance=.05,
                                  pick_depart_distance=.05,
                                  pick_depart_direction=rm.const.z_ax,
                                  use_rrt=True)


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


anime_data = Data(mot_data)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mot_data):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    mesh_model.show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
