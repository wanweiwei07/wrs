from wrs import wd, rm, rtqhe, mgm, mcm, gpa, gg

base = wd.World(cam_pos=[1.7, -1, 1.2], lookat_pos=[0, 0, .3])
# mgm.gen_frame().attach_to(base)
obj_cmodel = mcm.CollisionModel("tiltedlshape.stl")
obj_cmodel.rgba = rm.vec(.7, .7, 0, 1)
obj_cmodel.attach_to(base)

gripper = rtqhe.RobotiqHE()
grasp_collection = gpa.plan_gripper_grasps(gripper=gripper,
                                           obj_cmodel=obj_cmodel,
                                           angle_between_contact_normals=rm.radians(180),
                                           rotation_interval=rm.radians(10),
                                           max_samples=30,
                                           min_dist_between_sampled_contact_points=.001,
                                           contact_offset=.001,
                                           toggle_dbg=False)
grasp_collection.save_to_disk(file_name="robotiqhe_grasps.pickle")
grasp_collection = gg.GraspCollection.load_from_disk(file_name="robotiqhe_grasps.pickle")
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos,jaw_center_rotmat=grasp.ac_rotmat,jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.3).attach_to(base)
base.run()
