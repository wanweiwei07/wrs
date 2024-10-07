from wrs import rm, rtq85, mcm, mgm, wd, gpa, gag
import pickle

base = wd.World(cam_pos=rm.vec(.3, .3, .3), lookat_pos=rm.vec(0, 0, 0))
mgm.gen_frame(ax_length=.05, ax_radius=.0021).attach_to(base)
# object
object_bunny = mcm.CollisionModel("objects/bunnysim.stl")
object_bunny.rgb = rm.vec(.9, .75, .35)
object_bunny.attach_to(base)
# # contact pairs
# contact_pairs, contact_points = gpa.plan_contact_pairs(object_bunny,
#                                                        max_samples=10000,
#                                                        min_dist_between_sampled_contact_points=.014,
#                                                        angle_between_contact_normals=rm.radians(160),
#                                                        toggle_sampled_points=True)
# for p in contact_points:
#     mgm.gen_sphere(p, radius=.002).attach_to(base)
# pickle.dump(contact_pairs, open( "save.p", "wb" ))
# base.run()
contact_pairs = pickle.load(open("save.p", "rb"))
for i, cp in enumerate(contact_pairs):
    contact_p0, contact_n0 = cp[0]
    contact_p1, contact_n1 = cp[1]
    rgb = rm.const.tab20_list[i % 20]
    mgm.gen_sphere(contact_p0, radius=.002, rgb=rgb).attach_to(base)
    mgm.gen_arrow(contact_p0, contact_p0 + contact_n0 * .02, stick_radius=.0012, rgb=rgb).attach_to(base)
    # mgm.gen_arrow(contact_p0, contact_p0-contact_n0*.1, major_radius=.0012, rgb=rgb).attach_to(base)
    mgm.gen_sphere(contact_p1, radius=.002, rgb=rgb).attach_to(base)
    # mgm.gen_dashstick(contact_p0, contact_p1, major_radius=.0012, rgb=rgb).attach_to(base)
    mgm.gen_arrow(contact_p1, contact_p1 + contact_n1 * .02, stick_radius=.0012, rgb=rgb).attach_to(base)
    # mgm.gen_dasharrow(contact_p1, contact_p1+contact_n1*.03, major_radius=.0012, rgb=rgb).attach_to(base)
base.run()
gripper = rtq85.Robotiq85()
contact_offset = .002
grasp_collection = gag.GraspCollection(end_effector=gripper)
for i, cp in enumerate(contact_pairs):
    print(f"{i} of {len(contact_pairs)} done!")
    contact_p0, contact_n0 = cp[0]
    contact_p1, contact_n1 = cp[1]
    contact_center = (contact_p0 + contact_p1) / 2
    jaw_width = rm.norm(contact_p0 - contact_p1) + contact_offset * 2
    if jaw_width > gripper.jaw_range[1]:
        continue
    hndy = contact_n0
    hndz = rm.orthogonal_vector(contact_n0)
    grasp_collection += gag.define_gripper_grasps_with_rotation(gripper, object_bunny,
                                                               jaw_center_pos=contact_center,
                                                               approaching_direction=hndz,
                                                               thumb_opening_direction=hndy,
                                                               jaw_width=jaw_width,
                                                               rotation_interval=rm.radians(30),
                                                               toggle_flip=True)
for grasp in grasp_collection:
    gripper.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat, jaw_width=grasp.ee_values)
    gripper.gen_meshmodel(alpha=.7).attach_to(base)
base.run()
