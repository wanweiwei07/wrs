import os, copy, time
from wrs import wd, rm, gg, gpa, cbt, cbt_g, hop, mcm

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

cbt1 = cbt.Cobotta(pos=rm.vec(0, -.15, 0), rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.radians(90)), name="cbt1")
cbt2 = cbt.Cobotta(pos=rm.vec(0, .15, 0), rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.radians(-90)), name="cbt2")

cbt1.gen_meshmodel().attach_to(base)
cbt2.gen_meshmodel().attach_to(base)

gripper = cbt_g.CobottaGripper()

mesh_name = "bracket"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name + ".stl")
grasp_path = os.path.join(os.getcwd(), "pickles", gripper.name + mesh_name + "_grasp.pickle")
hopgcollection_path = os.path.join(os.getcwd(), "pickles", "cbt" + mesh_name + "_regspot.pickle")

obj_cmodel = mcm.CollisionModel(mesh_path)

sender_reference_grasps = gg.GraspCollection.load_from_disk(file_name=grasp_path)
receiver_reference_grasps = copy.deepcopy(sender_reference_grasps)

hopg_collection = hop.HOPGCollection(obj_cmodel=obj_cmodel, sender_robot=cbt1, receiver_robot=cbt2,
                                     sender_reference_gc=sender_reference_grasps,
                                     receiver_reference_gc=receiver_reference_grasps)
hopg_collection.add_new_hop(pos=rm.vec(0, 0, .4), rotmat=rm.rotmat_from_euler(rm.pi/2,0,0))
hopg_collection.save_to_disk(file_name=hopgcollection_path)

hopg_collection = hopg_collection.load_from_disk(file_name=hopgcollection_path)
meshmodel_list = hopg_collection.gen_meshmodel()


class Data(object):
    def __init__(self, meshmodel_list):
        self.counter = 0
        self.meshmodel_list = meshmodel_list


anime_data = Data(meshmodel_list=meshmodel_list)

print(len(anime_data.meshmodel_list))


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.meshmodel_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.meshmodel_list):
        anime_data.counter = 0
    anime_data.meshmodel_list[anime_data.counter].attach_to(base)
    if base.inputmgr.keymap['space']:
        print(anime_data.counter)
        anime_data.counter += 1
        time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
