from wrs import wd, rm, gpa, rtq85, mcm, mgm

base = wd.World(cam_pos=rm.vec(1, 1, 1), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame().attach_to(base)
# object
# object_box = mcm.gen_box(xyz_lengths=[.02, .06, 1])
# object_box.set_rgba([.7, .5, .3, .7])
# object_box.attach_to(base)
object_bunny = mcm.CollisionModel("objects/bunnysim.stl")
object_bunny.rgba = rm.vec(.9, .75, .35, 1)
object_bunny.attach_to(base)
# grippers
gripper = rtq85.Robotiq85()
# gripper.gen_meshmodel(toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
# base.run()

grasp_collection = gpa.plan_gripper_grasps(gripper, object_bunny, min_dist_between_sampled_contact_points=.01)
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.3).attach_to(base)
base.run()