from wrs import wd, rm, mgm, mcm, cbt, gg, ppp, rrtc

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=1)
ground.pos = rm.np.array([0, 0, -.5])
ground.attach_to(base)
ground.show_cdprim()
## object holder
holder_1 = mcm.CollisionModel("objects/holder.stl")
holder_1.rgba = rm.np.array([.5, .5, .5, 1])
h1_gl_pos = rm.np.array([-.15, -.2, .0])
h1_gl_rotmat = rm.rotmat_from_euler(0, 0, rm.pi)
holder_1.pos = h1_gl_pos
holder_1.rotmat = h1_gl_rotmat
mgm.gen_frame().attach_to(holder_1)
# visualize a copy
h1_copy = holder_1.copy()
h1_copy.attach_to(base)
h1_copy.show_cdprim()
## object holder goal
holder_2 = mcm.CollisionModel("objects/holder.stl")
h2_gl_pos = rm.np.array([.2, -.12, .0])
h2_gl_rotmat = rm.rotmat_from_euler(0, 0, rm.pi / 3)
holder_2.pos = h2_gl_pos
holder_2.rotmat = h2_gl_rotmat
# visualize a copy
h2_copy = holder_2.copy()
h2_copy.rgb = rm.const.tab20_list[0]
h2_copy.alpha = .3
h2_copy.attach_to(base)
h2_copy.show_cdprim()
## cobotta
robot = cbt.Cobotta()
robot.gen_meshmodel().attach_to(base)
robot.cc.show_cdprim()
# base.run()

rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

grasp_collection = gg.GraspCollection.load_from_disk(file_name='cobotta_gripper_grasps.pickle')
start_conf = robot.get_jnt_values()

goal_pose_list = [(h2_gl_pos, h2_gl_rotmat)]
mot_data = ppp.gen_pick_and_place(obj_cmodel=holder_1,
                                  end_jnt_values=start_conf,
                                  grasp_collection=grasp_collection,
                                  goal_pose_list=goal_pose_list,
                                  place_approach_distance_list=[.05] * len(goal_pose_list),
                                  place_depart_distance_list=[.05] * len(goal_pose_list),
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
