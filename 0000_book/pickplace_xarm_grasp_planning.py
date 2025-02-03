from wrs import wd, rm, x6wg2, mgm, mcm, gpa

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
object_box = mcm.gen_box(xyz_lengths=[.02, .04, .7])
object_box.rgba = rm.vec(.7, .5, .3, .7)
object_box.attach_to(base)
# base.run()
# gripper
gripper = x6wg2.end_effector.WRSGripper2()
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                           object_box,
                                           angle_between_contact_normals=rm.radians(175),
                                           rotation_interval=rm.radians(15),
                                           max_samples=5,
                                           min_dist_between_sampled_contact_points=.001,
                                           contact_offset=.001,
                                           toggle_dbg=False)
print(grasp_collection)
grasp_collection.save_to_disk(file_name="wg2_long_box.pickle")
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.1).attach_to(base)
base.run()