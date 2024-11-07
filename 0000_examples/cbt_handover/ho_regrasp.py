import os, copy
from wrs import wd, rm, mcm, mgm, cbt, cbt_g, gg, hop, horeg

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
regrasp_planner = horeg.HandoverPlanner(obj_cmodel=obj_cmodel, sender_robot=cbt1, receiver_robot=cbt2,
                                        sender_reference_grasps=sender_reference_grasps,
                                        receiver_reference_grasps=receiver_reference_grasps)
regrasp_planner.add_hopg_collection_from_disk(hopgcollection_path)
regrasp_planner.show_graph()
