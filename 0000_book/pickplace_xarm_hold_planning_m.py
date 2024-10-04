import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import math
import numpy as np
from wrs import basis as rm, robot_sim as xsm, robot_sim as xag, motion as adp, motion as rrtc, modeling as gm, \
    modeling as cm

base = wd.World(cam_pos=[2, -2, 2], lookat_pos=[.0, 0,.3])
gm.gen_frame().attach_to(base)

ground = cm.gen_box(xyz_lengths=[5, 5, 1], rgba=[.57, .57, .5, .7])
ground.set_pos(np.array([0,0,-.5]))
ground.attach_to(base)

object_box = cm.gen_box(xyz_lengths=[.02, .06, .7], rgba=[.7, .5, .3, .7])
object_box_gl_pos = np.array([.5,-.3,.35])
object_box_gl_rotmat = np.eye(3)
object_box.set_pos(object_box_gl_pos)
object_box.set_rotmat(object_box_gl_rotmat)
gm.gen_frame().attach_to(object_box)

robot_s = xsm.XArmShuidi()
rrtc_s = rrtc.RRTConnect(robot_s)
adp_s = adp.ADPlanner(robot_s)

grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_long_box.pickle')
component_name = "arm"

gripper_s = xag.XArmGripper()
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gl_jaw_center_pos = object_box_gl_pos+object_box_gl_rotmat.dot(jaw_center_pos)
    gl_jaw_center_rotmat = object_box_gl_rotmat.dot(jaw_center_rotmat)
    conf_path, jw_path = adp_s.gen_approach(component_name,
                                            gl_jaw_center_pos,
                                            gl_jaw_center_rotmat,
                                            start_conf=robot_s.get_jnt_values(component_name),
                                            linear_direction=gl_jaw_center_rotmat[:, 2],
                                            linear_distance=.2)
    if conf_path is None:
        continue
    else:
        robot_s.fk(component_name, conf_path[-1])
        rel_obj_pos, rel_obj_rotmat = robot_s.hold(object_box, jawwidth=jaw_width)
        # robot_s.gen_meshmodel().attach_to(base)
        # base.run()
        goal_jaw_center_pos = gl_jaw_center_pos+np.array([-.3,-.2,.3])
        goal_jaw_center_rotmat = rm.rotmat_from_euler(math.pi/2,math.pi/2,math.pi/2)
        jvs = robot_s.ik(component_name,
                         goal_jaw_center_pos,
                         goal_jaw_center_rotmat)
        robot_s.fk(component_name, jvs)
        # robot_s.gen_meshmodel().attach_to(base)
        # base.run()
        path = rrtc_s.plan(component_name,
                           start_conf=conf_path[-1],
                           goal_conf=jvs,
                           obstacle_list=[],
                           ext_dist=.1, rand_rate=70, max_time=300)
        for jvp in path:
            robot_s.fk(component_name, jvp)
            gl_obj_pos, gl_obj_rotmat = robot_s.cvt_loc_tcp_to_gl(component_name, rel_obj_pos, rel_obj_rotmat)
            robot_s.gen_meshmodel().attach_to(base)
        break
base.run()