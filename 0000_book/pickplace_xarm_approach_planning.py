import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import numpy as np
from wrs import robot_sim as xsm, robot_sim as xag, motion as adp, modeling as gm, modeling as cm

base = wd.World(cam_pos=[1.5, -.5, 2], lookat_pos=[.3, -.03,.05])
gm.gen_frame().attach_to(base)

ground = cm.gen_box(xyz_lengths=[5, 5, 1], rgba=[.57, .57, .5, .7])
ground.set_pos(np.array([0,0,-.5]))
ground.attach_to(base)

object_box = cm.gen_box(xyz_lengths=[.02, .06, .2], rgba=[.7, .5, .3, .7])
object_box_gl_pos = np.array([.5,-.3,.1])
object_box_gl_rotmat = np.eye(3)
object_box.set_pos(object_box_gl_pos)
object_box.set_rotmat(object_box_gl_rotmat)
gm.gen_frame().attach_to(object_box)
object_box.attach_to(base)
object_box.show_cdprim()

robot_s = xsm.XArmShuidi()
robot_s.gen_meshmodel().attach_to(base)
adp_s = adp.ADPlanner(robot_s)

grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_box.pickle')
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
        for jvs in conf_path:
            robot_s.fk(component_name, jvs)
            robot_s.gen_meshmodel(rgba=[0,1,1,.3]).attach_to(base)
        break
base.run()