import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import grasping.annotation.utils as gu
import pickle

base = wd.World(cam_pos=[.3, .3, .3], lookat_pos=[0, 0, 0])
gm.gen_frame(length=.05, thickness=.0021).attach_to(base)
# object
object_bunny = cm.CollisionModel("objects/bunnysim.stl")
object_bunny.set_rgba([.9, .75, .35, .3])
object_bunny.attach_to(base)
# hnd_s
# contact_pairs, contact_points = gpa.plan_contact_pairs(object_bunny,
#                                                        max_samples=10000,
#                                                        min_dist_between_sampled_contact_points=.014,
#                                                        angle_between_contact_normals=math.radians(160),
#                                                        toggle_sampled_points=True)
# for p in contact_points:
#     gm.gen_sphere(p, radius=.002).attach_to(base)
# base.run()
# pickle.dump(contact_pairs, open( "save.p", "wb" ))
contact_pairs = pickle.load(open( "save.p", "rb" ))
for i, cp in enumerate(contact_pairs):
    contact_p0, contact_n0 = cp[0]
    contact_p1, contact_n1 = cp[1]
    rgba = rm.get_rgba_from_cmap(i)
    gm.gen_sphere(contact_p0, radius=.002, rgba=rgba).attach_to(base)
    gm.gen_arrow(contact_p0, contact_p0+contact_n0*.01, thickness=.0012, rgba = rgba).attach_to(base)
    # gm.gen_arrow(contact_p0, contact_p0-contact_n0*.1, thickness=.0012, rgba = rgba).attach_to(base)
    gm.gen_sphere(contact_p1, radius=.002, rgba=rgba).attach_to(base)
    # gm.gen_dashstick(contact_p0, contact_p1, thickness=.0012, rgba=rgba).attach_to(base)
    gm.gen_arrow(contact_p1, contact_p1+contact_n1*.01, thickness=.0012, rgba=rgba).attach_to(base)
    # gm.gen_dasharrow(contact_p1, contact_p1+contact_n1*.03, thickness=.0012, rgba=rgba).attach_to(base)
# base.run()
gripper_s = rtq85.Robotiq85()
contact_offset = .002
grasp_info_list = []
for i, cp in enumerate(contact_pairs):
    print(f"{i} of {len(contact_pairs)} done!")
    contact_p0, contact_n0 = cp[0]
    contact_p1, contact_n1 = cp[1]
    contact_center = (contact_p0 + contact_p1) / 2
    jaw_width = np.linalg.norm(contact_p0 - contact_p1) + contact_offset * 2
    if jaw_width > gripper_s.jawwidth_rng[1]:
        continue
    hndy = contact_n0
    hndz = rm.orthogonal_vector(contact_n0)
    grasp_info_list += gu.define_grasp_with_rotation(gripper_s,
                                                     object_bunny,
                                                     gl_jaw_center_pos=contact_center,
                                                     gl_jaw_center_z=hndz,
                                                     gl_jaw_center_y=hndy,
                                                     jaw_width=jaw_width,
                                                     gl_rotation_ax=hndy,
                                                     rotation_interval=math.radians(30),
                                                     toggle_flip=True)
for grasp_info in grasp_info_list:
    aw_width, gl_jaw_center, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.jaw_to(aw_width)
    gripper_s.gen_meshmodel().attach_to(base)
base.run()