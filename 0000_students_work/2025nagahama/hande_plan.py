from wrs import wd, rm, rtqhe, mgm, mcm, gpa, gg

base = wd.World(cam_pos=[1.7, -1, 1.2], lookat_pos=[0, 0, .3])
# mgm.gen_frame().attach_to(base)
obj_cmodel = mcm.CollisionModel("lshape.stl")
obj_cmodel.rgba = rm.vec(.7, .7, 0, 1)
obj_cmodel.attach_to(base)

gripper = rtqhe.RobotiqHE()
# grasp_collection = gpa.plan_gripper_grasps(gripper=gripper,
#                                            obj_cmodel=obj_cmodel,
#                                            angle_between_contact_normals=rm.radians(180),
#                                            rotation_interval=rm.radians(15),
#                                            max_samples=5,
#                                            min_dist_between_sampled_contact_points=.001,
#                                            contact_offset=.001,
#                                            toggle_dbg=False)
# grasp_collection.save_to_disk(file_name="robotiqhe_grasps.pickle")
grasp_collection = gg.GraspCollection.load_from_disk(file_name="robotiqhe_grasps.pickle")
grasp = grasp_collection[0]
gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
gripper.gen_meshmodel(alpha=1).attach_to(base)
grasp = grasp_collection[20]
gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
gripper.gen_meshmodel(alpha=1).attach_to(base)
grasp = grasp_collection[50]
gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
gripper.gen_meshmodel(alpha=1).attach_to(base)

# for i in range(100):
#     randomness = .02*rm.np.random.randn(3)
#     randomness[1]=0
#     obj_cmodel_copy = obj_cmodel.copy()
#     obj_cmodel_copy.pos = obj_cmodel.pos + randomness
#     obj_cmodel_copy.alpha = .3
#     if gripper.is_mesh_collided(obj_cmodel_copy):
#         continue
#     obj_cmodel_copy.attach_to(base)
#
# mgm.gen_box(xyz_lengths=rm.vec(.3, 0.001, .3), pos = grasp.ac_pos, rotmat = grasp.ac_rotmat, alpha=.3).attach_to(base)
base.run()
