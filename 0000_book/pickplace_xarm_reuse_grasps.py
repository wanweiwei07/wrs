import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import numpy as np
import robot_sim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xsm
import robot_sim.end_effectors.grippers.xarm_gripper.xarm_gripper as xag

base = wd.World(cam_pos=[3, 3, 0.5], lookat_pos=[0, 0, 0.4])
gm.gen_frame().attach_to(base)

ground = cm.gen_box(extent=[5, 5, 1], rgba=[.57, .57, .5, .7])
ground.set_pos(np.array([0,0,-.5]))
ground.attach_to(base)

object_box = cm.gen_box(extent=[.02, .06, .2], rgba=[.7, .5, .3, .7])
object_box_gl_pos = np.array([.5,-.3,.1])
object_box_gl_rotmat = np.eye(3)
object_box.set_pos(object_box_gl_pos)
object_box.set_rotmat(object_box_gl_rotmat)
gm.gen_frame().attach_to(object_box)
object_box.attach_to(base)

robot_s = xsm.XArm7YunjiMobile()
robot_s.gen_meshmodel().attach_to(base)

grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_box.pickle')
component_name = "arm"

gripper_s = xag.XArmGripper()
for grasp_info in grasp_info_list:
    jaw_width, jaw_center, hnd_pos, hnd_rotmat = grasp_info
    gl_jaw_center_pos = object_box_gl_pos+object_box_gl_rotmat.dot(jaw_center)
    gl_jaw_center_rotmat = object_box_gl_rotmat.dot(hnd_rotmat)
    jnt_values =  robot_s.ik(component_name,
                             tgt_pos=gl_jaw_center_pos,
                             tgt_rotmat=gl_jaw_center_rotmat)
    gripper_s.grip_at(gl_jaw_center_pos, gl_jaw_center_rotmat[:,2], gl_jaw_center_rotmat[:,1], jaw_width)
    gripper_s.gen_meshmodel().attach_to(base)
    # if jnt_values is not None:
    #     gripper_s.grip_at(gl_jaw_center_pos, gl_jaw_center_rotmat[:,2], gl_jaw_center_rotmat[:,1], jaw_width)
    #     gripper_s.gen_meshmodel(rgba=[.7,.7,.7,.3]).attach_to(base)
    #     robot_s.fk(component_name,
    #                jnt_values=jnt_values)
    #     robot_s.gen_meshmodel(rgba=[0,1,0,.3]).attach_to(base)
    #     break
base.run()