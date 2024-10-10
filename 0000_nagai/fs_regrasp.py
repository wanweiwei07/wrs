import os
import time
import networkx as nx
from wrs import wd, rm, mcm, ko2fg, fsp, fsreg, gg

mesh_name = "bracketR1"
mesh_path = os.path.join(os.getcwd(), "meshes", mesh_name+".stl")
fsref_pose_path = os.path.join(os.getcwd(), "pickles", mesh_name+"_fsref_pose.pickle")
grasp_path = os.path.join(os.getcwd(), "pickles", mesh_name+"_grasp.pickle")
regspot_path = os.path.join(os.getcwd(), "pickles", mesh_name+"_regspot.pickle")

base = wd.World(cam_pos=rm.vec(1, 1, 1), lookat_pos=rm.vec(0, 0, 0))
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), pos=rm.vec(0, 0, -0.5))
ground.show_cdprim()
ground.attach_to(base)
bunny = mcm.CollisionModel(mesh_path)
robot = ko2fg.KHI_OR2FG7()

fs_reference_poses = fsp.FSReferencePoses.load_from_disk(file_name=fsref_pose_path)
reference_grasps = gg.GraspCollection.load_from_disk(file_name=grasp_path)
fsreg_planner = fsreg.FSRegraspPlanner(robot=robot,
                                       obj_cmodel=bunny,
                                       fs_reference_poses=fs_reference_poses,
                                       reference_grasp_collection=reference_grasps)
fsreg_planner.add_fsregspot_collection_from_disk(regspot_path)

start_node_list = fsreg_planner.add_start_pose(obj_pose=(rm.np.array([.5, -.3, .04]), rm.np.eye(3)),
                                               obstacle_list=[ground])
goal_node_list = fsreg_planner.add_goal_pose(
    obj_pose=(rm.np.array([-.3, -.5, .04]), rm.rotmat_from_euler(0, rm.pi, 0)), obstacle_list=[ground])

min_path = None
for start in start_node_list:
    for goal in goal_node_list:
        path = nx.shortest_path(fsreg_planner._graph, source=start, target=goal)
        min_path = path if min_path is None else path if len(path) < len(min_path) else min_path

print(min_path)
# fsreg_planner.show_graph_with_path(min_path)

mesh_model_list = fsreg_planner.gen_regrasp_motion(path=min_path, obstacle_list=[ground],
                                                   linear_distance=.15)

class Data(object):
    def __init__(self, mesh_model_list):
        self.counter = 0
        self.mesh_model_list = mesh_model_list


anime_data = Data(mesh_model_list=mesh_model_list)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mesh_model_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mesh_model_list):
        # for mesh_model in anime_data.mot_data.mesh_list:
        #     mesh_model.detach()
        anime_data.counter = 0
    anime_data.mesh_model_list[anime_data.counter].attach_to(base)
    anime_data.mesh_model_list[anime_data.counter].show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    # time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.1, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
