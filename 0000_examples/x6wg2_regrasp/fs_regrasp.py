import os
from wrs import wd, rm, mcm, x6wg2, fsp, fsreg, gg

mesh_name = "bunnysim"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name + ".stl")
fsref_pose_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_fsref_pose.pickle")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_grasp.pickle")
regspot_path = os.path.join(os.getcwd(), "pickles", mesh_name + "_regspot.pickle")

base = wd.World(cam_pos=rm.vec(1, 1, 1), lookat_pos=rm.vec(0, 0, 0))
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), pos=rm.vec(0, 0, -0.51))
ground.show_cdprim()
ground.attach_to(base)
obj = mcm.CollisionModel(mesh_path)
robot = x6wg2.XArmLite6WG2()

fs_reference_poses = fsp.FSReferencePoses.load_from_disk(file_name=fsref_pose_path)
reference_grasps = gg.GraspCollection.load_from_disk(file_name=grasp_path)
fsreg_planner = fsreg.FSRegraspPlanner(robot=robot,
                                       obj_cmodel=obj,
                                       fs_reference_poses=fs_reference_poses,
                                       reference_gc=reference_grasps)
fsreg_planner.add_fsregspot_collection_from_disk(regspot_path)

start_pose = (rm.np.array([.3, .1, 0]), rm.np.eye(3))
goal_pose = (rm.np.array([.3, -.3, 0]), rm.rotmat_from_euler(rm.pi / 3, rm.pi / 6, 0))
obj_start = obj.copy()
obj_start.rgb = rm.const.red
obj_start.pose = start_pose
obj_start.attach_to(base)
obj_goal = obj.copy()
obj_goal.rgb = rm.const.green
obj_goal.pose = goal_pose
obj_goal.attach_to(base)
result = fsreg_planner.plan_by_obj_poses(start_pose=start_pose, goal_pose=goal_pose, obstacle_list=[ground],
                                         linear_distance=.05, toggle_dbg=False)
if result is None:
    print("No solution found.")
    exit()

class Data(object):
    def __init__(self, mesh_model_list):
        self.counter = 0
        self.mesh_model_list = mesh_model_list


anime_data = Data(mesh_model_list=result.mesh_list)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mesh_model_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mesh_model_list):
        anime_data.counter = 0
    anime_data.mesh_model_list[anime_data.counter].attach_to(base)
    anime_data.mesh_model_list[anime_data.counter].show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
