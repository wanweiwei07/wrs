import os
import time

import numpy as np

from wrs import wd, rm, mcm, mgm, fsreg, ur3ed, gg, fsp

mesh_name = "bracketR1"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name + ".stl")
fsref_pose_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_fsref_pose.pickle")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_grasp.pickle")
regspot_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_regspot.pickle")

base = wd.World(cam_pos=[2, 1, 2], lookat_pos=[0, 0, 1])
# ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, .01), pos=rm.vec(0, 0, -0.005))
# ground.attach_to(base)
bunny = mcm.CollisionModel(mesh_path)
robot = ur3ed.UR3e_Dual()
robot.lft_arm.gen_meshmodel().attach_to(base)
robot.lft_arm.cc.show_cdprim()

fs_reference_poses = fsp.FSReferencePoses.load_from_disk(file_name=fsref_pose_path)
reference_grasps = gg.GraspCollection.load_from_disk(file_name=grasp_path)
fsregspot_collection = fsp.FSRegSpotCollection(robot=robot.lft_arm,
                                               obj_cmodel=bunny,
                                               fs_reference_poses=fs_reference_poses,
                                               reference_gc=reference_grasps)
spot_pos0 = rm.np.array([.8, .35, .78])
spot_pos1 = rm.np.array([.8, .45, .78])
spot_pos2 = rm.np.array([.8, .5, 1])
# mgm.gen_sphere(pos=spot_pos0).attach_to(base)
# mgm.gen_sphere(pos=spot_pos1).attach_to(base)
# mgm.gen_sphere(pos=spot_pos2).attach_to(base)
# base.run()
fsregspot_collection.add_new_spot(spot_pos=spot_pos0, barrier_z_offset=None, spot_rotz=np.radians(90))
fsregspot_collection.add_new_spot(spot_pos=spot_pos1, barrier_z_offset=None, spot_rotz=np.radians(90))
fsregspot_collection.add_new_spot(spot_pos=spot_pos2, barrier_z_offset=None, spot_rotz=np.radians(90))
fsregspot_collection.save_to_disk(regspot_path)
fsregspot_collection.load_from_disk(regspot_path)
mesh_model_list = fsregspot_collection.gen_meshmodel()
for fsregspot in fsregspot_collection:
    mcm.mgm.gen_frame(pos=fsregspot.pos,
                      rotmat=rm.rotmat_from_euler(0, 0, fsregspot.rotz)).attach_to(base)


class Data(object):
    def __init__(self, mesh_model_list):
        self.counter = 0
        self.mesh_model_list = mesh_model_list


anime_data = Data(mesh_model_list=mesh_model_list)

print(len(anime_data.mesh_model_list))


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mesh_model_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mesh_model_list):
        anime_data.counter = 0
    anime_data.mesh_model_list[anime_data.counter].attach_to(base)
    if base.inputmgr.keymap['space']:
        print(anime_data.counter)
        anime_data.counter += 1
        time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
