import os
import time
from wrs import wd, rm, mcm, mgm, reg, x6wg2, gg, fsp

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
obj_path = os.path.join("objects", "bunnysim.stl")
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, .01), pos=rm.vec(0, 0, -0.005))
ground.attach_to(base)
bunny = mcm.CollisionModel(obj_path)
robot = x6wg2.XArmLite6WG2()

reference_fsp_poses = fsp.ReferenceFSPPoses.load_from_disk(file_name="reference_fsp_poses_bunny.pickle")
reference_grasps = gg.GraspCollection.load_from_disk(file_name="reference_grasps_wg2_bunny.pickle")
regspot_collection = reg.RegraspSpotCollection(robot=robot,
                                               obj_cmodel=bunny,
                                               reference_fsp_poses=reference_fsp_poses,
                                               reference_grasp_collection=reference_grasps)
regspot_collection.add_new_spot(spot_pos=rm.np.array([.4, 0, 0]), spot_rotz=0)
regspot_collection.add_new_spot(spot_pos=rm.np.array([.4, .2, 0]), spot_rotz=0)
regspot_collection.add_new_spot(spot_pos=rm.np.array([.4, -.2, 0]), spot_rotz=0)
regspot_collection.save_to_disk("regspot_collection_x6wg2_bunny.pickle")
mesh_model_list = regspot_collection.gen_meshmodel()
for regspot in regspot_collection:
    mcm.mgm.gen_frame(pos=regspot.pos,
                      rotmat=rm.rotmat_from_euler(0, 0, regspot.rotz)).attach_to(base)


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
