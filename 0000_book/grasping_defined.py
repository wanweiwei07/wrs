import math
import numpy as np
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.annotation.utils as gu
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import basis.robot_math as rm

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object_box = cm.gen_box(xyz_lengths=[.02, .06, .1])
object_box.set_rgba([.7, .5, .3, .7])
object_box.attach_to(base)
# gripper
rtq85_s = rtq85.Robotiq85()
# rtq85_s.gen_meshmodel().attach_to(base)
# base.run()
# gl_approaching_direction = rm.rotmat_from_axangle(np.array([0,1,0]), -math.pi/6).dot(np.array([-1,0,0]))
# grasp_info_list = gau.define_grasp(rtq85_s, object_box,
#                                   jaw_center_pos=np.array([0,0,0]),
#                                   gl_approaching_direction=gl_approaching_direction,
#                                   gl_fgr1_opening_direction=np.array([0,1,0]),
#                                   jaw_width=.065,
#                                   toggle_flip=True,
#                                   toggle_dbg=True)
grasp_info_list = gu.define_gripper_grasps_with_rotation(rtq85_s, object_box, gl_jaw_center_pos=np.array([0, 0, 0]),
                                                         gl_approaching_vec=np.array([-1, 0, 0]),
                                                         gl_fgr0_opening_vec=np.array([0, 1, 0]), jaw_width=.065,
                                                         rotation_interval=math.radians(30),
                                                         rotation_range=(math.radians(-180), math.radians(180)),
                                                         toggle_flip=False, toggle_debug=True)
# for grasp_info in grasp_info_list:
#     aw_width, jaw_center_pos, jaw_center_rotmat, gripper_root_pos, gripper_root_rotmat = grasp_info
#     rtq85_s.fix_to(gripper_root_pos, gripper_root_rotmat)
#     rtq85_s.jaw_to(aw_width)
#     rtq85_s.gen_meshmodel().attach_to(base)
# mgm.gen_frame(pos=gripper_root_pos, rotmat=gripper_root_rotmat).attach_to(base)
base.run()
