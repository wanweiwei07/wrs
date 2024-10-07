from wrs import rm, rtq85, mcm, mgm, wd, gag

base = wd.World(cam_pos=rm.vec(1, 1, 1), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)
# object
object_box = mcm.gen_box(xyz_lengths=rm.vec(.02, .06, .1))
object_box.rgb = rm.vec(.7, .5, .3)
object_box.attach_to(base)
# grippers
gripper = rtq85.Robotiq85()
grasp_collection = gag.define_gripper_grasps_with_rotation(gripper, object_box,
                                                           jaw_center_pos=rm.vec(0, 0, 0),
                                                           approaching_direction=rm.vec(-1, 0, 0),
                                                           thumb_opening_direction=rm.vec(0, 1, 0),
                                                           jaw_width=.065,
                                                           rotation_interval=rm.radians(30),
                                                           rotation_range=(rm.radians(-180), rm.radians(180)),
                                                           toggle_flip=False)
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.7).attach_to(base)
    mgm.gen_frame(pos=gripper.pos, rotmat=gripper.rotmat).attach_to(base)
base.run()
