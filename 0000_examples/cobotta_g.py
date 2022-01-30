import visualization.panda.world as wd
import grasping.planning.antipodal as gp
import robot_sim.end_effectors.gripper.cobotta_gripper.cobotta_gripper as cg
import modeling.collision_model as cm
import modeling.geometric_model as gm
import numpy as np
import math

base = wd.World(cam_pos=np.array([.5, .5, .5]), lookat_pos=np.array([0, 0, 0]))
gm.gen_frame().attach_to(base)
objcm = cm.CollisionModel("objects/holder.stl")
objcm.attach_to(base)
# base.run()

hnd_s = cg.CobottaGripper()
# hnd_s.gen_meshmodel().attach_to(base)
# base.run()
grasp_info_list = gp.plan_grasps(hnd_s,
                                 objcm,
                                 angle_between_contact_normals=math.radians(175),
                                 openning_direction='loc_y',
                                 rotation_interval=math.radians(15),
                                 max_samples=20,
                                 min_dist_between_sampled_contact_points=.001,
                                 contact_offset=.001)
gp.write_pickle_file(objcm_name="holder",
                     grasp_info_list=grasp_info_list,
                     file_name="cobg_holder_grasps.pickle")
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    hnd_s.grip_at_with_jcpose(jaw_center_pos, jaw_center_rotmat, jaw_width)
    hnd_s.gen_meshmodel().attach_to(base)
base.run()