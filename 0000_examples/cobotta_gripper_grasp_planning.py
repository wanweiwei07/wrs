import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
from wrs import robot_sim as cg, modeling as mcm
import numpy as np
import math

base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
# mgm.gen_frame().attach_to(base)
obj_cmodel = mcm.CollisionModel("objects/holder.stl")
obj_cmodel.attach_to(base)

gripper = cg.CobottaGripper()
# grippers.gen_meshmodel().attach_to(base)
# base.run()
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                          obj_cmodel,
                                          angle_between_contact_normals=math.radians(175),
                                          rotation_interval=math.radians(15),
                                          max_samples=20,
                                          min_dist_between_sampled_contact_points=.001,
                                          contact_offset=.001,
                                          toggle_dbg=False)
print(grasp_collection)
# gpa.write_pickle_file(obj_name="holder",
#                       grasp_info_list=grasp_info_list,
#                       file_name="cobotta_gripper_grasps.pickle")
for grasp_info in grasp_collection:
    # jaw_width, jaw_center_pos, jaw_center_rotmat, gripper_root_pos, gripper_root_rotmat = grasp_info
    # grippers.grip_at_by_pose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    gripper.grip_at_by_pose(grasp_info.ac_pos, grasp_info.ac_rotmat, grasp_info.ee_values)
    gripper.gen_meshmodel(alpha=1).attach_to(base)
base.run()