import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import math
import numpy as np
from wrs import basis as rm, robot_sim as cbt, manipulation as ppp, motion as rrtc, modeling as mgm, modeling as mcm

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=np.array([5, 5, 1]), rgb=np.array([.7, .7, .7]), alpha=1)
ground.pos = np.array([0, 0, -.51])
ground.attach_to(base)
# object holder
holder_1 = mcm.CollisionModel("objects/holder.stl")
holder_1.rgba = np.array([.5, .5, .5, 1])
h1_gl_pos = np.array([-.15, -.3, .0])
h1_gl_rotmat = rm.rotmat_from_euler(0, 0, math.pi / 2)
holder_1.pos = h1_gl_pos
holder_1.rotmat = h1_gl_rotmat

mgm.gen_frame().attach_to(holder_1)
h1_copy = holder_1.copy()
h1_copy.attach_to(base)

# object holder goal
holder_2 = mcm.CollisionModel("objects/holder.stl")
h2_gl_pos = np.array([.3, -.15, .05])
h2_gl_rotmat = rm.rotmat_from_euler(0, 0, 2*math.pi / 3)
holder_2.pos = h2_gl_pos
holder_2.rotmat = h2_gl_rotmat

h2_copy = holder_2.copy()
h2_copy.rgb = rm.bc.tab20_list[0]
h2_copy.alpha = .3
h2_copy.attach_to(base)

robot = cbt.Cobotta()
robot.gen_meshmodel().attach_to(base)

rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

original_grasp_info_list = gpa.load_pickle_file(obj_name='holder', path='./', file_name='cobg_holder_grasps.pickle')
start_conf = robot.get_jnt_values()
print(original_grasp_info_list)

mot_data = ppp.gen_pick_and_place(obj_cmodel=holder_1,
                                  grasp_info_list=original_grasp_info_list,
                                  end_jnt_values=start_conf,
                                  goal_pose_list=[(h2_gl_pos, h2_gl_rotmat)])


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
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
