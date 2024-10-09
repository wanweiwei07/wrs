import os
import time
import networkx as nx
from wrs import wd, rm, mcm, x6wg2, fsp, fsreg, gg

base = wd.World(cam_pos=rm.vec(1, 1, 1), lookat_pos=rm.vec(0, 0, 0))
obj_path = os.path.join("objects", "bunnysim.stl")
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), pos=rm.vec(0, 0, -0.5))
ground.show_cdprim()
ground.attach_to(base)
bunny = mcm.CollisionModel(obj_path)
robot = x6wg2.XArmLite6WG2()

fs_reference_poses = fsp.FSReferencePoses.load_from_disk(file_name="fs_reference_poses_bunny.pickle")
reference_grasps = gg.GraspCollection.load_from_disk(file_name="reference_grasps_wg2_bunny.pickle")
fsreg_planner = fsreg.FSRegraspPlanner(robot=robot,
                                       obj_cmodel=bunny,
                                       fs_reference_poses=fs_reference_poses,
                                       reference_grasp_collection=reference_grasps)
fsreg_planner.add_fsregspot_collection_from_disk("regspot_collection_x6wg2_bunny.pickle")

start_node_list = fsreg_planner.add_start_pose(obj_pose=(rm.np.array([.2, .2, 0]), rm.np.eye(3)),
                                               obstacle_list=[ground])
goal_node_list = fsreg_planner.add_goal_pose(
    obj_pose=(rm.np.array([.2, -.2, 0]), rm.rotmat_from_euler(rm.pi / 3, rm.pi / 6, 0)), obstacle_list=[ground])

min_path = None
for start in start_node_list:
    for goal in goal_node_list:
        path = nx.shortest_path(fsreg_planner._graph, source=start, target=goal)
        min_path = path if min_path is None else path if len(path) < len(min_path) else min_path

print(min_path)
# fsreg_planner.show_graph_with_path(min_path)

mesh_model_list = fsreg_planner.gen_regrasp_motion(path=min_path, obstacle_list=[ground])


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
