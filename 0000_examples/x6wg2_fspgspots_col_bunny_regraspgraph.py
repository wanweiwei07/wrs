import os
import time
import basis.robot_math as rm
import networkx as nx
import numpy as np
import visualization.panda.world as wd
import modeling.collision_model as mcm
import manipulation.placement.flat_surface_placement as mpfsp
import grasping.grasp as g
import manipulation.regrasp as reg
import manipulation.regrasp as rg
import robot_sim.robots.xarmlite6_wg.x6wg2 as x6wg2

if __name__ == '__main__':
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    obj_path = os.path.join("objects", "bunnysim.stl")
    ground = mcm.gen_box(xyz_lengths=[5, 5, .01], pos=np.array([0, 0, -0.005]))
    ground.attach_to(base)
    bunny = mcm.CollisionModel(obj_path)
    robot = x6wg2.XArmLite6WG2()

    reference_fsp_poses = mpfsp.ReferenceFSPPoses.load_from_disk(file_name="reference_fsp_poses_bunny.pickle")
    reference_grasps = g.GraspCollection.load_from_disk(file_name="reference_wg2_bunny_grasps.pickle")
    fsreg_planner = reg.FSRegraspPlanner(robot=robot,
                                         obj_cmodel=bunny,
                                         reference_fsp_poses=reference_fsp_poses,
                                         reference_grasp_collection=reference_grasps)
    fsreg_planner.add_regspot_col_from_disk("regspot_col_x6wg2_bunny.pickle")

    start_node_list = fsreg_planner.add_start_pose(obj_pose=(np.array([.4, .2, 0]), np.eye(3)))
    goal_node_list = fsreg_planner.add_goal_pose(
        obj_pose=(np.array([.4, -.2, 0]), rm.rotmat_from_euler(np.pi / 3, np.pi / 6, 0)))

    min_path = None
    for start in start_node_list:
        for goal in goal_node_list:
            path = nx.shortest_path(fsreg_planner.fsreg_graph, source=start, target=goal)
            min_path = path if min_path is None else path if len(path) < len(min_path) else min_path

    # fsreg_planner.draw_fsreg_graph()
    # fsreg_planner.draw_path(min_path)

    mesh_model_list = fsreg_planner.gen_regrasp_motion(path=min_path)


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
        # if base.inputmgr.keymap['space']:
        #     print(anime_data.counter)
        anime_data.counter += 1
            # time.sleep(.5)
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

    base.run()
