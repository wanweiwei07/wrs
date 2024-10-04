import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import math
import numpy as np
from wrs import basis as rm, robot_sim as xsm, manipulation as ppp, motion as rrtc, modeling as gm, modeling as cm

base = wd.World(cam_pos=[2.1, -2.1, 2.1], lookat_pos=[.0, 0, .3])
gm.gen_frame().attach_to(base)
# ground
ground = cm.gen_box(xyz_lengths=[5, 5, 1], rgba=[.57, .57, .5, .7])
ground.set_pos(np.array([0, 0, -.51]))
ground.attach_to(base)
# object box
object_box = cm.gen_box(xyz_lengths=[.02, .06, .7], rgba=[.7, .5, .3, .7])
# object_box_gl_pos = np.array([.3, -.4, .35])
# object_box_gl_rotmat = np.eye(3)
object_box_gl_pos = np.array([.6, .2, .35])
object_box_gl_rotmat = rm.rotmat_from_euler(0,0,math.pi/2)
obgl_start_homomat = rm.homomat_from_posrot(object_box_gl_pos, object_box_gl_rotmat)
object_box.set_pos(object_box_gl_pos)
object_box.set_rotmat(object_box_gl_rotmat)
gm.gen_frame().attach_to(object_box)
object_box_copy = object_box.copy()
object_box_copy.attach_to(base)
# object box goal
# object_box_gl_goal_pos = np.array([.6, -.1, .1])
# object_box_gl_goal_rotmat = rm.rotmat_from_euler(0, math.pi / 2, math.pi / 2)
object_box_gl_goal_pos = np.array([.35, -.35, .01])
object_box_gl_goal_rotmat = rm.rotmat_from_euler(math.pi/3, math.pi / 2, math.pi / 2)
obgl_goal_homomat = rm.homomat_from_posrot(object_box_gl_goal_pos, object_box_gl_goal_rotmat)
object_box_goal_copy = object_box.copy()
object_box_goal_copy.set_homomat(obgl_goal_homomat)
object_box_goal_copy.attach_to(base)

robot_s = xsm.XArmShuidi()
robot_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
# base.run()
rrtc_s = rrtc.RRTConnect(robot_s)
ppp_s = ppp.PickPlacePlanner(robot_s)

original_grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_long_box.pickle')
hnd_name = "hnd"
start_conf = robot_s.get_jnt_values(hnd_name)
conf_list, jawwidth_list, objpose_list = \
    ppp_s.gen_pick_and_place_motion(hnd_name=hnd_name,
                                    objcm=object_box,
                                    grasp_info_list=original_grasp_info_list,
                                    start_conf=start_conf,
                                    end_conf=start_conf,
                                    goal_homomat_list=[obgl_start_homomat, obgl_goal_homomat],
                                    approach_direction_list=[None, np.array([0, 0, -1])],
                                    approach_distance_list=[.2] * 2,
                                    depart_direction_list=[np.array([0, 0, 1]), None],
                                    depart_distance_list=[.2] * 2)
robot_attached_list = []
object_attached_list = []
counter = [0]
def update(robot_s,
           hnd_name,
           object_box,
           robot_path,
           jawwidth_path,
           obj_path,
           robot_attached_list,
           object_attached_list,
           counter,
           task):
    if counter[0] >= len(robot_path):
        counter[0] = 0
    if len(robot_attached_list) != 0:
        for robot_attached in robot_attached_list:
            robot_attached.detach()
        for object_attached in object_attached_list:
            object_attached.detach()
        robot_attached_list.clear()
        object_attached_list.clear()
    pose = robot_path[counter[0]]
    robot_s.fk(hnd_name, pose)
    robot_s.change_jaw_width(hnd_name, jawwidth_path[counter[0]])
    robot_meshmodel = robot_s.gen_mesh_model()
    robot_meshmodel.attach_to(base)
    robot_attached_list.append(robot_meshmodel)
    obj_pose = obj_path[counter[0]]
    objb_copy = object_box.copy()
    objb_copy.set_homomat(obj_pose)
    objb_copy.attach_to(base)
    object_attached_list.append(objb_copy)
    counter[0] += 1
    return task.again
taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot_s,
                                 hnd_name,
                                 object_box,
                                 conf_list,
                                 jawwidth_list,
                                 objpose_list,
                                 robot_attached_list,
                                 object_attached_list,
                                 counter],
                      appendTask=True)
base.run()
