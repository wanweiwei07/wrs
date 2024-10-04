import math
import numpy as np
import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
from wrs import robot_sim as yg, modeling as mgm, modeling as mcm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
# tube
obj_cmodel = mcm.CollisionModel("objects/tubebig.stl")
obj_cmodel.rgba = np.array([.9, .75, .35, 1])
obj_cmodel.attach_to(base)
# grippers
grpr = yg.YumiGripper()
grasp_info_list = gpa.plan_gripper_grasps(grpr, obj_cmodel,
                                          angle_between_contact_normals=math.radians(177),
                                          max_samples=15, min_dist_between_sampled_contact_points=.005,
                                          contact_offset=.005)
gpa.write_pickle_file('tubebig', grasp_info_list, './', 'yumi_tube_big.pickle')
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    grpr.grip_at_by_pose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    grpr.gen_meshmodel(alpha=.3).attach_to(base)
base.run()