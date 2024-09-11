import visualization.panda.world as wd
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.wrs_gripper as wg
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import modeling.collision_model as mcm
import numpy as np
import math
import bench_mark.ycb as ycb

base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
obj_cmodel = mcm.CollisionModel(ycb.washer_10mm_file)
obj_cmodel.attach_to(base)
base.run()
# mgm.gen_frame().attach_to(base)
for file in ycb.all_files:
    obj_cmodel = mcm.CollisionModel(file)
    obj_cmodel.attach_to(base)

    # gripper = wg.WRSGripper3()
    gripper = rtq85.Robotiq85()
    # gripper.gen_meshmodel().attach_to(base)
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
#     gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
#     gripper.gen_meshmodel(alpha=.1).attach_to(base)
base.run()
