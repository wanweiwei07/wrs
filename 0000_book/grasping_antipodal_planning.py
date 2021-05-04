import math
import numpy as np
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import robot_sim.grippers.robotiq85.robotiq85 as rtq85

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
gm.gen_frame().attach_to(base)
# object
object_box = cm.gen_box(extent=[.02, .06, 1])
object_box.set_rgba([.7, .5, .3, .7])
object_box.attach_to(base)
# hnd_s
gripper_s = rtq85.Robotiq85()
grasp_info_list = gpa.plan_grasps(gripper_s, object_box, max_samples=5)
for grasp_info in grasp_info_list:
    aw_width, gl_jaw_center, hnd_pos, hnd_rotmat = grasp_info
    gripper_s.fix_to(hnd_pos, hnd_rotmat)
    gripper_s.jaw_to(aw_width)
    gripper_s.gen_meshmodel(rgba=[0,1,0,.3]).attach_to(base)
base.run()
