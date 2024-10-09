import os
import time
from wrs import wd, rm, mcm, mgm, fsreg, x6wg2, gg, fsp

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
obj_path = os.path.join("objects", "bunnysim.stl")
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, .01), pos=rm.vec(0, 0, -0.005))
ground.attach_to(base)
bunny = mcm.CollisionModel(obj_path)
robot = x6wg2.XArmLite6WG2()

fs_reference_poses = fsp.FSReferencePoses.load_from_disk(file_name="fs_reference_poses_bunny.pickle")
reference_grasps = gg.GraspCollection.load_from_disk(file_name="reference_grasps_wg2_bunny.pickle")
fsregspot_collection = fsreg.FSRegSpotCollection(robot=robot,
                                                 obj_cmodel=bunny,
                                                 fs_reference_poses=fs_reference_poses,
                                                 reference_grasp_collection=reference_grasps)
fsregspot_collection.add_new_spot(spot_pos=rm.np.array([.4, 0, 0]), spot_rotz=0)
fsregspot_collection.add_new_spot(spot_pos=rm.np.array([.4, .2, 0]), spot_rotz=0)
fsregspot_collection.add_new_spot(spot_pos=rm.np.array([.4, -.2, 0]), spot_rotz=0)
fsregspot_collection.save_to_disk("regspot_collection_x6wg2_bunny.pickle")
fsregspot_collection.load_from_disk("regspot_collection_x6wg2_bunny.pickle")
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
