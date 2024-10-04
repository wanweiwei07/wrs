import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
from wrs import robot_sim as rtq85, bench_mark as ycb, modeling as mcm
import numpy as np
import math

base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
# mgm.gen_frame().attach_to(base)
for file in ycb.all_files.values():
    obj_cmodel = mcm.CollisionModel(file)
    obj_cmodel.attach_to(base)

    # grippers = wg.WRSGripper3()
    gripper = rtq85.Robotiq85()
    # grippers.gen_meshmodel().attach_to(base)
    # base.run()
    grasp_collection = gpa.plan_gripper_grasps(gripper,
                                               obj_cmodel,
                                               angle_between_contact_normals=math.radians(175),
                                               rotation_interval=math.radians(30),
                                               max_samples=100,
                                               min_dist_between_sampled_contact_points=.005,
                                               contact_offset=.001,
                                               toggle_dbg=False)
    print(file, len(grasp_collection))
# # print(grasp_collection)
# # grasp_collection.save_to_disk(file_name="wrs_gripper2_grasps.pickle")
# print(len(grasp_collection))
# for grasp in grasp_collection:
#     grippers.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
#     grippers.gen_meshmodel(alpha=.1).attach_to(base)
base.run()
