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
                                        sender_reference_gc=sender_reference_grasps,
                                        receiver_reference_gc=receiver_reference_grasps)
regrasp_planner.add_hopg_collection_from_disk(hopgcollection_path)

start_pose = (rm.np.array([.15, .15, .04]), rm.np.eye(3))
goal_pose = (rm.np.array([-.15, -.15, .04]), rm.rotmat_from_euler(0, rm.pi, 0))
obj_start = obj_cmodel.copy()
obj_start.rgb = rm.const.red
obj_start.pose = start_pose
obj_start.attach_to(base)
obj_goal = obj_cmodel.copy()
obj_goal.rgb = rm.const.green
obj_goal.pose = goal_pose
obj_goal.attach_to(base)

regrasp_planner.plan_by_obj_poses(start_pose=start_pose, goal_pose=goal_pose)
# # base.run()
# regrasp_planner.sender_fsreg_planner.add_start_pose(obj_pose=start_pose)
# regrasp_planner.receiver_fsreg_planner.add_start_pose(obj_pose=start_pose)
# regrasp_planner.sender_fsreg_planner.add_goal_pose(obj_pose=goal_pose)
# regrasp_planner.receiver_fsreg_planner.add_goal_pose(obj_pose=goal_pose)
#
# regrasp_planner.show_graph()
