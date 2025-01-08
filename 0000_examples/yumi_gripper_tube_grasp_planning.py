from wrs import wd, rm, gg, yumi_g, mcm, gpa

base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
# mgm.gen_frame().attach_to(base)
obj_cmodel = mcm.CollisionModel("objects/tubebig.stl")
obj_cmodel.attach_to(base)

gripper = yumi_g.YumiGripper()
# gripper.gen_meshmodel().attach_to(base)
# base.run()
grasp_collection = gpa.plan_gripper_grasps(gripper,
                                           obj_cmodel,
                                           angle_between_contact_normals=rm.radians(175),
                                           rotation_interval=rm.radians(15),
                                           max_samples=20,
                                           min_dist_between_sampled_contact_points=.001,
                                           contact_offset=.001,
                                           toggle_dbg=False)
grasp_collection.save_to_disk(file_name="yumi_gripper_tube_big.pickle")
for grasp in grasp_collection:
    gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
    gripper.gen_meshmodel(alpha=.1).attach_to(base)
base.run()
