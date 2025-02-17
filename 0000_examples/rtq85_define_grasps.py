from wrs import wd, rm, rtq85, mcm, gag

base = wd.World(cam_pos=[.5, .5, .3], lookat_pos=[0, 0, 0])
gripper = rtq85.Robotiq85()
obj_cm = mcm.CollisionModel("./objects/bunnysim.stl")
obj_cm.pos = rm.vec(.0, -.0065, -0.012)
obj_cm.attach_to(base)
obj_cm.show_local_frame()
grasp_collection = gag.define_gripper_grasps_with_rotation(gripper, obj_cm, jaw_center_pos=rm.vec(0, 0, 0),
                                                           approaching_direction=rm.vec(1, 0, 0),
                                                           thumb_opening_direction=rm.vec(0, 1, 0), jaw_width=.055)
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                            jaw_width=grasp.ee_values)
    gripper.gen_meshmodel().attach_to(base)
base.run()
