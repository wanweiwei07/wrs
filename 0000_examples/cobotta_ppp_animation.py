import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import grasping.planning.antipodal as gpa
import math
import numpy as np
import basis.robot_math as rm
import robot_sim.robots.cobotta.cobotta as cbt
import manipulation.pick_place_planner as ppp
import motion.probabilistic.rrt_connect as rrtc

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
gm.gen_frame().attach_to(base)
# ground
ground = cm.gen_box(extent=[5, 5, 1], rgba=[.7, .7, .7, .7])
ground.set_pos(np.array([0, 0, -.51]))
ground.attach_to(base)
# object holder
object_holder = cm.CollisionModel("objects/holder.stl")
object_holder.set_rgba([.5,.5,.5,1])
object_holder_gl_pos = np.array([-.15, -.3, .0])
object_holder_gl_rotmat = np.eye(3)
obgl_start_homomat = rm.homomat_from_posrot(object_holder_gl_pos, object_holder_gl_rotmat)
object_holder.set_pos(object_holder_gl_pos)
object_holder.set_rotmat(object_holder_gl_rotmat)
gm.gen_frame().attach_to(object_holder)
object_holder_copy = object_holder.copy()
object_holder_copy.attach_to(base)
# object holder goal
object_holder_gl_goal_pos = np.array([.25, -.05, .05])
object_holder_gl_goal_rotmat = rm.rotmat_from_euler(0, 0, -math.pi / 2)
obgl_goal_homomat = rm.homomat_from_posrot(object_holder_gl_goal_pos, object_holder_gl_goal_rotmat)
object_holder_goal_copy = object_holder.copy()
object_holder_goal_copy.set_homomat(obgl_goal_homomat)
object_holder_goal_copy.attach_to(base)

robot_s = cbt.Cobotta()
# robot_s.gen_meshmodel().attach_to(base)
# base.run()
rrtc_s = rrtc.RRTConnect(robot_s)
ppp_s = ppp.PickPlacePlanner(robot_s)

original_grasp_info_list = gpa.load_pickle_file('holder', './', 'cobg_holder_grasps.pickle')
manipulator_name = "arm"
hand_name = "hnd"
start_conf = robot_s.get_jnt_values(manipulator_name)
conf_list, jawwidth_list, objpose_list = \
    ppp_s.gen_pick_and_place_motion(hnd_name=hand_name,
                                    objcm=object_holder,
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
    robot_s.fk(manipulator_name, pose)
    robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    robot_attached_list.append(robot_meshmodel)
    obj_pose = obj_path[counter[0]]
    objb_copy = object_box.copy()
    objb_copy.set_rgba([1,0,0,1])
    objb_copy.set_homomat(obj_pose)
    objb_copy.attach_to(base)
    object_attached_list.append(objb_copy)
    counter[0] += 1
    return task.again

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot_s,
                                 object_holder,
                                 conf_list,
                                 jawwidth_list,
                                 objpose_list,
                                 robot_attached_list,
                                 object_attached_list,
                                 counter],
                      appendTask=True)
base.run()
