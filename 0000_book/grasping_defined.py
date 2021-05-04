import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.annotation.utils as gu
import robot_sim.grippers.robotiq85.robotiq85 as rtq85

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object_box = cm.gen_box(extent=[.02, .06, .1])
object_box.set_rgba([.7, .5, .3, .7])
object_box.attach_to(base)
# hnd_s
rtq85_s = rtq85.Robotiq85()
# rtq85_s.gen_meshmodel().attach_to(base)
# base.run()
# gl_hndz = rm.rotmat_from_axangle(np.array([0,1,0]), -math.pi/6).dot(np.array([-1,0,0]))
# grasp_info_list = gu.define_grasp(rtq85_s, object_box,
#                                   gl_jaw_center=np.array([0,0,0]),
#                                   gl_hndz=gl_hndz,
#                                   gl_hndy=np.array([0,1,0]),
#                                   jaw_width=.065,
#                                   toggle_flip=True)
grasp_info_list = gu.define_grasp_with_rotation(rtq85_s, object_box,
                                                gl_jaw_center=np.array([0, 0, 0]),
                                                gl_hndz=np.array([-1, 0, 0]),
                                                gl_hndy=np.array([0, 1, 0]),
                                                jaw_width=.065,
                                                gl_rotation_ax=np.array([0, 1, 0]),
                                                rotation_range=(math.radians(-180), math.radians(180)),
                                                rotation_interval=math.radians(30),
                                                toggle_flip=False,
                                                toggle_debug=True)
# for grasp_info in grasp_info_list:
#     aw_width, gl_jaw_center, hnd_pos, hnd_rotmat = grasp_info
#     rtq85_s.fix_to(hnd_pos, hnd_rotmat)
#     rtq85_s.jaw_to(aw_width)
#     rtq85_s.gen_meshmodel().attach_to(base)
# gm.gen_frame(pos=hnd_pos, rotmat=hnd_rotmat).attach_to(base)
base.run()
