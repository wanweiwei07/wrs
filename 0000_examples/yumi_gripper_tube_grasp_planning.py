import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
from wrs import robot_sim as yg, modeling as mcm
import numpy as np
import math

base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
# mgm.gen_frame().attach_to(base)
obj_cmodel = mcm.CollisionModel("objects/tubebig.stl")
obj_cmodel.attach_to(base)

gripper = yg.YumiGripper()
# grippers.gen_meshmodel().attach_to(base)
# base.run()
grasp_info_list = gpa.plan_gripper_grasps(gripper,
                                          obj_cmodel,
                                          angle_between_contact_normals=math.radians(175),
                                          rotation_interval=math.radians(15),
                                          max_samples=20,
                                          min_dist_between_sampled_contact_points=.001,
                                          contact_offset=.001,
                                          toggle_dbg=False)
print(grasp_info_list)
gpa.write_pickle_file(obj_name="tubebig",
                      grasp_info_list=grasp_info_list,
                      file_name="yumi_gripper_tube_big.pickle")
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, gripper_root_pos, gripper_root_rotmat = grasp_info
    gripper.grip_at_by_pose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper.gen_meshmodel(alpha=.1).attach_to(base)
base.run()