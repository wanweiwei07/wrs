import os, copy
from wrs import wd, rm, mcm, mgm, cbt, cbt_g, gg, hop, horeg

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

sender_cbt = cbt.Cobotta(pos=rm.vec(0, -.15, 0), rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.radians(90)), name="cbt1")
receiver_cbt = cbt.Cobotta(pos=rm.vec(0, .15, 0), rotmat=rm.rotmat_from_axangle(rm.const.z_ax, rm.radians(-90)), name="cbt2")

# sender_cbt.gen_meshmodel().attach_to(base)
# receiver_cbt.gen_meshmodel().attach_to(base)

gripper = cbt_g.CobottaGripper()

mesh_name = "bracket"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name + ".stl")
grasp_path = os.path.join(os.getcwd(), "pickles", gripper.name + mesh_name + "_grasp.pickle")
hopgcollection_path = os.path.join(os.getcwd(), "pickles", "cbt" + mesh_name + "_regspot.pickle")

obj_cmodel = mcm.CollisionModel(mesh_path)

sender_reference_grasps = gg.GraspCollection.load_from_disk(file_name=grasp_path)
receiver_reference_grasps = copy.deepcopy(sender_reference_grasps)
regrasp_planner = horeg.HandoverPlanner(obj_cmodel=obj_cmodel, sender_robot=sender_cbt, receiver_robot=receiver_cbt,
                                        sender_reference_gc=sender_reference_grasps,
                                        receiver_reference_gc=receiver_reference_grasps)
regrasp_planner.add_hopg_collection_from_disk(hopgcollection_path)

start_pose = (rm.np.array([.15, -.15, .04]), rm.np.eye(3))
goal_pose = (rm.np.array([-.15, .15, .04]), rm.rotmat_from_euler(0, rm.pi, 0))
obj_start = obj_cmodel.copy()
obj_start.rgb = rm.const.red
obj_start.pose = start_pose
obj_start.attach_to(base)
obj_goal = obj_cmodel.copy()
obj_goal.rgb = rm.const.green
obj_goal.pose = goal_pose
obj_goal.attach_to(base)

# base.run()

result = regrasp_planner.plan_by_obj_poses(start_pose=start_pose, goal_pose=goal_pose)
if result is None:
    print("No solution found.")
    exit()


class Data(object):
    def __init__(self, mesh_list):
        self.counter = 0
        self.mesh_list = mesh_list

mesh_list = []
for sgl_result in result:
    mesh_list += sgl_result.mesh_list
anime_data = Data(mesh_list=mesh_list)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mesh_list):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    anime_data.mesh_list[anime_data.counter].attach_to(base)
    anime_data.mesh_list[anime_data.counter].show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    # time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()