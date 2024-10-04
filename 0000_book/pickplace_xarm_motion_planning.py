import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import numpy as np
from wrs import robot_sim as xsm, robot_sim as xag, motion as rrtc, modeling as gm, modeling as cm
import wrs.motion.optimization_based.incremental_nik as inik

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
rrtc_s = rrtc.RRTConnect(robot_s)
inik_s = inik.IncrementalNIK(robot_s)

grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_box.pickle')
component_name = "arm"

gripper_s = xag.XArmGripper()
cnter = 0
for grasp_info in grasp_info_list:
    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
    gl_jaw_center_pos = object_box_gl_pos+object_box_gl_rotmat.dot(jaw_center_pos)
    gl_jaw_center_rotmat = object_box_gl_rotmat.dot(jaw_center_rotmat)
    gripper_s.grip_at_by_pose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)
    if not gripper_s.is_mesh_collided([ground]):
        retracted_gl_jaw_center_pos=gl_jaw_center_pos-gl_jaw_center_rotmat[:,2]*.2
        gripper_s.grip_at_by_pose(retracted_gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)
        if not gripper_s.is_mesh_collided([ground]):
            jnt_values =  robot_s.ik(component_name,
                                     tgt_pos=gl_jaw_center_pos,
                                     tgt_rotmat=gl_jaw_center_rotmat)
            retracted_jnt_values = robot_s.ik(component_name,
                                              tgt_pos=retracted_gl_jaw_center_pos,
                                              tgt_rotmat=gl_jaw_center_rotmat)
            if jnt_values is None or retracted_jnt_values is None:
                pass
            else:
                if cnter == 6:
                    gripper_s.gen_meshmodel(rgba=[0,0,1,.3]).attach_to(base)
                    # robot_s.fk(hnd_name, jnt_values=jnt_values)
                    # robot_s.gen_meshmodel(rgba=[0,0,1,.3]).attach_to(base)
                    # robot_s.show_cdprimit()
                    # base.run()
                    path = rrtc_s.plan(component_name,
                                       start_conf=robot_s.get_jnt_values(component_name),
                                       goal_conf=retracted_jnt_values,
                                       obstacle_list=[object_box],
                                       ext_dist=.1,
                                       max_time=300)
                    for jvp in path:
                        robot_s.fk(component_name, jvp)
                        robot_s.gen_meshmodel(rgba=[1,0,1,.3]).attach_to(base)
                    linear_path = inik_s.gen_linear_motion(component_name,
                                                           retracted_gl_jaw_center_pos,
                                                           gl_jaw_center_rotmat,
                                                           gl_jaw_center_pos,
                                                           gl_jaw_center_rotmat)
                    for jvp in linear_path:
                        robot_s.fk(component_name, jvp)
                        robot_s.gen_meshmodel(rgba=[0,1,1,.3]).attach_to(base)
                    break
                cnter+=1
    else:
        pass
base.run()