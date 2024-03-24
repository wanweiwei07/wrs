import grasping.grasp
import visualization.panda.world as wd
import modeling.geometric_model as mgm
import modeling.collision_model as mcm
import math
import time
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.xarmlite6_wg.x6wg2 as x6g2
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc
import grasping.grasp as gg
import manipulation.placement as mpl

base = wd.World(cam_pos=[1, .0, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# robot
robot = x6g2.XArmLite6WG2()
# table
table_obstacle = mcm.gen_surface_barrier()
table_obstacle.attach_to(base)
# object
bunny = mcm.CollisionModel("objects/bunnysim.stl")
# grasp_collection = gg.GraspCollection.load_from_disk(file_name="grasps_wg2_bunny.pickle")
#
# tgt
tgt_xyz= np.array([.4, -.05, 0])
#
# reference_fsp_pose_list, reference_support_facet_list = mpl.get_reference_fsp_pose_list(obj_cmodel=bunny,
#                                                                                         stability_threshhold=.1,
#                                                                                         toggle_support_facets=True)
# fspg_col = mpl.FSPGCollection(reference_fsp_pose_list=reference_fsp_pose_list,
#                               reference_grasp_collection=grasp_collection)
# fspg_col += mpl.get_fspg_collection_at_given_tgt_spot(robot=robot,
#                                                       reference_fsp_pose_list=reference_fsp_pose_list,
#                                                       reference_grasp_collection=grasp_collection,
#                                                       tgt_xyz=tgt_xyz,
#                                                       consider_robot=True,
#                                                       toggle_dbg=False)
# fspg_col.save_to_disk("x6wg2_bunny_fspg_col.pickle")
# base.run()


# goal spot frame
mgm.gen_frame(pos=tgt_xyz).attach_to(base)

fspg_col = mpl.FSPGCollection.load_from_disk("x6wg2_bunny_fspg_col.pickle")

mesh_model_list = mpl.gen_meshmodel_list(robot=robot, obj_cmodel=bunny, fspg_col=fspg_col)
print(fspg_col)


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
    if base.inputmgr.keymap['space']:
        print(anime_data.counter)
        anime_data.counter += 1
        time.sleep(.5)
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
