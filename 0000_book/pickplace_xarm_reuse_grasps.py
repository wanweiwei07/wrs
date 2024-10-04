import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import numpy as np
from wrs import robot_sim as xsm, robot_sim as xag, modeling as gm, modeling as cm

base = wd.World(cam_pos=[3, 3, 0.5], lookat_pos=[0, 0, 0.4])
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

robot_s = xsm.XArmShuidi()
robot_s.gen_meshmodel().attach_to(base)
base.run()

grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_box.pickle')
component_name = "arm"

gripper_s = xag.XArmGripper()
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gl_jaw_center_pos = object_box_gl_pos+object_box_gl_rotmat.dot(jaw_center_pos)
    gl_jaw_center_rotmat = object_box_gl_rotmat.dot(jaw_center_rotmat)
    gl_hnd_pos = object_box_gl_pos+object_box_gl_rotmat.dot(hnd_pos)
    gl_hnd_rotmat = object_box_gl_rotmat.dot(hnd_rotmat)
    gripper_s.fix_to(gl_hnd_pos, gl_hnd_rotmat)
    if not gripper_s.is_mesh_collided([ground]):
        jnt_values =  robot_s.ik(component_name,
                                 tgt_pos=gl_jaw_center_pos,
                                 tgt_rotmat=gl_jaw_center_rotmat)
        if jnt_values is None:
            gripper_s.gen_meshmodel(rgba=[0,1,0,.3]).attach_to(base)
        else:
            gripper_s.gen_meshmodel(rgba=[0,0,1,.3]).attach_to(base)
            robot_s.fk(component_name, jnt_values=jnt_values)
            robot_s.gen_meshmodel(rgba=[0,0,1,.3]).attach_to(base)
    else:
        gripper_s.gen_meshmodel(rgba=[1,0,0,.3]).attach_to(base)
base.run()