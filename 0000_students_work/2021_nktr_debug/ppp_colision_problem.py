import wrs.visualization.panda.world as wd
import wrs.grasping.planning.antipodal as gpa
import numpy as np
from wrs import basis as rm, robot_sim as nxt, robot_sim as xsm, manipulation as ppp, motion as rrtc, modeling as gm, \
    modeling as cm
import math
import wrs.motion.probabilistic.rrt_differential_wheel_connect as rrtdwc

base = wd.World(cam_pos=[2, -2, 2], lookat_pos=[.0, 0, .3])
gm.gen_frame().attach_to(base)
ground = cm.gen_box(xyz_lengths=[5, 5, 1], rgba=[.57, .57, .5, .7])
ground.set_pos(np.array([0, 0, -.5]))
ground.attach_to(base)
## show human
human = nxt.Nextage()
rotmat = rm.rotmat_from_axangle([0, 0, 1], math.pi/2)
human.fix_to(np.array([0, -1.5, 1]), rotmat=rotmat)
gm.gen_arrow(spos=np.array([0, -1.5, 1]),
             epos=np.array([0, -1.5, 0]),
             stick_radius=0.01,
             rgba=np.array([.5, 0, 0, 1])).attach_to(base)
human.gen_stickmodel().attach_to(base)
## show table2
table2_center = np.array([-1.3, .6, .483])
table2_bottom = cm.gen_box(xyz_lengths=[1.08, .42, .06], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
rotmat = rm.rotmat_from_axangle([0, 0, 1], math.pi/2)
table2_bottom.set_rotmat(rotmat)
table2_bottom.set_pos(table2_center - np.array([0, 0, .03]))
table2_bottom.attach_to(base)
table2_bottom2 = table2_bottom.copy()
table2_bottom2.set_pos(table2_center + np.array([0, 0, -.483 + .03]))
table2_bottom2.attach_to(base)
table2_top = table2_bottom.copy()
table2_top.set_pos(table2_center + np.array([0, 0, -.483 + 1 - .03]))
table2_top.attach_to(base)
table2_leg1 = cm.gen_box(xyz_lengths=[0.06, 0.06, 1], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table2_leg1.set_pos(table2_center + np.array([-.21 + .03, .54 - .03, -.483 + .5]))
table2_leg1.attach_to(base)
table2_leg2 = cm.gen_box(xyz_lengths=[0.06, 0.06, 1], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table2_leg2.set_pos(table2_center + np.array([.21 - .03, .54 - .03, -.483 + .5]))
table2_leg2.attach_to(base)
table2_leg3 = cm.gen_box(xyz_lengths=[0.06, 0.06, 1], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table2_leg3.set_pos(table2_center + np.array([-.21 + .03, -.54 + .03, -.483 + .5]))
table2_leg3.attach_to(base)
table2_leg4 = cm.gen_box(xyz_lengths=[0.06, 0.06, 1], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table2_leg4.set_pos(table2_center + np.array([.21 - .03, -.54 + .03, -.483 + .5]))
table2_leg4.attach_to(base)
## show worktable
table_center = np.array([0, -1, 0.8])
table_plate = cm.gen_box(xyz_lengths=[1.82, .8, .05], rgba=[.7, .5, .3, 1])
table_plate.set_pos(table_center - np.array([0, 0, .025]))
table_plate.attach_to(base)
table_leg1 = cm.gen_box(xyz_lengths=[0.03, 0.03, .8], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table_leg1.set_pos(table_center + np.array([.91 - .015, -.4 - .015, -.4]))
table_leg1.attach_to(base)
table_leg2 = cm.gen_box(xyz_lengths=[0.03, 0.03, .8], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table_leg2.set_pos(table_center + np.array([.91 - .015, .4 + .015, -.4]))
table_leg2.attach_to(base)
table_leg3 = cm.gen_box(xyz_lengths=[0.03, 0.03, .8], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table_leg3.set_pos(table_center + np.array([-.91 + .015, -.4 - .015, -.4]))
table_leg3.attach_to(base)
table_leg4 = cm.gen_box(xyz_lengths=[0.03, 0.03, .8], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
table_leg4.set_pos(table_center + np.array([-.91 + .015, .4 + .015, -.4]))
table_leg4.attach_to(base)
## show object
height = .018/2
object_box1 = cm.gen_box(xyz_lengths=[.296, .92, .018], rgba=[.7, .5, .3, 1])   # no.3
object_box1.set_pos(table2_center + np.array([0, 0, height]))
object_box1.attach_to(base)
height += .018
object_box2 = cm.gen_box(xyz_lengths=[.296, .92, .017], rgba=[.7, .5, .3, 1])   # no.4
object_box2.set_pos(table2_center + np.array([0, 0, height]))
object_box2.attach_to(base)
height += .017
object_box3 = cm.gen_box(xyz_lengths=[.296, .88, .018], rgba=[.7, .5, .3, 1])   # no.2
object_box3.set_pos(table2_center + np.array([0, 0, height]))
object_box3.attach_to(base)
object_box3.show_local_frame()
pre_homo = object_box3.get_homomat()
height += .018
object_box4 = cm.gen_box(xyz_lengths=[.296, .88, .018], rgba=[.7, .5, .3, .5])   # no.1
object_box4.set_pos(table2_center + np.array([0, 0, height]))
## show tool table
tool_table = cm.gen_box(xyz_lengths=[.3, .2, .3], rgba=[150 / 255, 154 / 255, 152 / 255, 1])
tool_table_pos = np.array([table2_leg2.get_pos()[0] + .15 + .03, table2_leg2.get_pos()[1] - .1 + .03, .15])
tool_table.set_pos(tool_table_pos)
tool_table.attach_to(base)
# set obstacles
obstacle_list = [tool_table, table2_top, table2_bottom, table2_bottom2, table2_leg1, table2_leg2, table2_leg3, table2_leg4,
                table_plate, table_leg1, table_leg2, table_leg3, table_leg4, object_box1, object_box2, object_box3, object_box4]
for i in obstacle_list:
    i.show_cdprim()
# base.run()
## show tool
tool = cm.gen_box(xyz_lengths=[.05, .2, .05])
tool_initial = tool.copy()
tool_final = tool.copy()
tool_initial.set_rgba([0, 0, 1, 1])
tool_initial_pos = tool_table_pos + np.array([0, 0, .15 + .03])
tool_initial_rotmat = rm.rotmat_from_axangle([0, 0, 1], math.radians(90))
tool_initial.set_pos(tool_initial_pos)
tool_initial.set_rotmat(tool_initial_rotmat)
# tool_initial.attach_to(base)
tool_initial.show_local_frame()
tool_initial_homomat = rm.homomat_from_posrot(tool_initial_pos, rotmat)
tool_final.set_rgba([0, 0, 1, 1])
tool_final_pos = np.array([table2_center[0], table2_center[1], object_box4.get_pos()[2] + .009 + .11])
rotmat = np.dot(rm.rotmat_from_axangle([0, 1, 0], math.radians(90)), rm.rotmat_from_axangle([0, 0, 1], math.radians(90)))
tool_final.set_pos(tool_final_pos)
tool_final.set_rotmat(rotmat)
tool_final_homomat = rm.homomat_from_posrot(tool_final_pos, rotmat)
## rbt_s and grasp list
robot_s = xsm.XArmShuidi()
rrtc_s = rrtc.RRTConnect(robot_s)
rrtc_planner = rrtdwc.RRTDWConnect(robot_s)
grasp_info_list = gpa.load_pickle_file('box', './', 'xarm_long_box.pickle')
grasp_info_list_tool = gpa.load_pickle_file('box', './', 'xarm_tool.pickle')
jnt_values_initial = robot_s.get_jnt_values(component_name="all")
jnt_values = jnt_values_initial.copy()
jnt_values[0] = tool_final_pos[0] + .8
jnt_values[1] = tool_final_pos[1]
jnt_values[2] = math.pi
robot_s.fk(component_name="all", jnt_values=np.array(jnt_values))
print(jnt_values_initial, jnt_values)
# robot_s.gen_meshmodel().attach_to(base)
## discretize the location and check if the place is available or not
# hnd_name = "hnd"
# ppp_s = ppp.PickPlacePlanner(robot_s)
# goal_homomat_list = [tool_initial_homomat, tool_final_homomat]
# target_obj = object_box4.copy()
# c_pos = target_obj.get_pos()
# c_pos[2] = 0
# dist_pos = []
# for j in range(3, 13):
#     s_pos = np.array([0, -.1 * j, 0])
#     for i in range(0, 10):
#         print("number", i)
#         rotmat = rm.rotmat_from_axangle(axis=np.array([0, 0, 1]), angle=math.radians(180 * i / 10))
#         pos = np.dot(rotmat, s_pos) + c_pos
#         jnt_values = pos.copy()
#         jnt_values[2] = math.pi
#         robot_s.fk(component_name="agv", jnt_values=jnt_values)
#         # check if the rbt_s is collided or not at the place
#         if robot_s.is_collided(obstacle_list=obstacle_list):
#             print("1")
#             continue
#         else:
#             print("2")
#             counter = 0
#             for grasp_info in grasp_info_list_tool:
#                 ee_values, jaw_center_pos, jaw_center_rotmat, gripper_root_pos, gripper_root_rotmat = grasp_info
#                 first_jaw_center_pos = tool_initial_rotmat.dot(jaw_center_pos) + tool_initial_pos
#                 first_jaw_center_rotmat = tool_initial_rotmat.dot(jaw_center_rotmat)
#                 armjnts = robot_s.ik(component_name="arm", tgt_pos=first_jaw_center_pos, tgt_rotmat=first_jaw_center_rotmat)
#                 if armjnts is not None:
#                     robot_s.fk(component_name="arm", jnt_values=armjnts)
#                     is_robot_collided = robot_s.is_collided(obstacle_list=obstacle_list)
#                     # display only if armjnts is not None and rbt_s is not collided
#                     if is_robot_collided is False and counter is 0:
#                         dist_pos.append(jnt_values)
#                         counter += 1
#                         continue
# print(dist_pos)


# gpa.write_pickle_file('box', dist_pos, './', 'dist_pos.pickle')

dist_pos = gpa.load_pickle_file('box', './', 'dist_pos.pickle')
for i in dist_pos:
    pos = i.copy()
    pos[2] = 0
    gm.gen_sphere(pos=pos).attach_to(base)


## ppp (tool ppp)
object_box = tool.copy()
target_obj = object_box4.copy()
hnd_name = "hnd"
ppp_s = ppp.PickPlacePlanner(robot_s)
start_conf = robot_s.get_jnt_values(component_name="arm")
goal_homomat_list = [tool_initial_homomat, tool_final_homomat]
for dict_goal_pos in dist_pos:
    print("move to the dist pos", dict_goal_pos)
    robot_s.fk(component_name="agv", jnt_values=dict_goal_pos)
    conf_list1, jawwidth_list1, objpose_list1 = \
        ppp_s.gen_pick_and_place_motion(hnd_name=hnd_name,
                                        objcm=tool,
                                        grasp_info_list=grasp_info_list_tool,
                                        start_conf=start_conf,
                                        end_conf=start_conf,
                                        goal_homomat_list=goal_homomat_list,
                                        approach_direction_list=[None, np.array([0, 0, -1])],
                                        approach_distance_list=[.1] * len(goal_homomat_list),
                                        depart_direction_list=[np.array([0, 0, 1]), None],
                                        obstacle_list=obstacle_list,
                                        depart_distance_list=[.1] * len(goal_homomat_list))
    if conf_list1 is not None:
        conf_list = robot_s.cvt_to_alljnts(conf_list1,hnd_name)
        jawwidth_list = jawwidth_list1
        objpose_list = objpose_list1
        objpose_list_box = []
        for i in range(len(conf_list)):
            objpose_list_box.append(target_obj.get_homomat())
        break
    #     ## cut department_conf
    #     conf_list2 = []
    #     jawwidth_list2 = []
    #     objpose_list2 = []
    #     obj_pos = target_obj.get_homomat()[:3, 3]
    #     obj_rotmat = target_obj.get_homomat()[:3, :3]
    #     objpose_list_box2 = ppp_s.gen_object_motion(component_name="arm", conf_list=conf_list1, obj_pos=obj_pos,
    #                                                 obj_rotmat=obj_rotmat, end_type="absolute")
    #     for i, objpose in enumerate(objpose_list1):
    #         objpose_list2.append(objpose)
    #         conf_list2.append(conf_list1[i])
    #         jawwidth_list2.append(jawwidth_list1[i])
    #         objpose2 = objpose.tolist()
    #         tool_final_homomat2 = tool_final_homomat.tolist()
    #         if objpose2 == tool_final_homomat2:
    #             break
    #     ## gen object sliding motion
    #     inik_s = inik.IncrementalNIK(robot_s)
    #     robot_s.fk(component_name="arm", jnt_values=conf_list2[-1])
    #     start_tcp_pos, start_tcp_rotmat = robot_s.get_gl_tcp(manipulator_name="arm")
    #     motion_vec = [1, 0, 0]
    #     linear_distance = .1
    #     goal_tcp_pos = start_tcp_pos + rm.unit_vector(motion_vec) * linear_distance
    #     goal_tcp_rotmat = start_tcp_rotmat
    #     linear_conf_list = inik_s.gen_linear_motion(component_name="arm", start_tcp_pos=start_tcp_pos,
    #                                                 start_tcp_rotmat=start_tcp_rotmat,
    #                                                 goal_tcp_pos=goal_tcp_pos, goal_tcp_rotmat=goal_tcp_rotmat,
    #                                                 granularity=0.01, seed_jnt_values=conf_list2[-1])
    #     ## if fail to plan sliding with arm ? then use agv
    #     if linear_conf_list is None:
    #         print("--- 1 ----")
    #         temp_linear_conf = []
    #         robot_s.fk(component_name="arm", jnt_values=conf_list2[-1])
    #         alljnts = robot_s.get_jnt_values(component_name="all")
    #         robot_pos = alljnts[0:3].tolist()
    #         for i in range(int(linear_distance / 0.01)):
    #             pos1 = robot_pos + rm.unit_vector(motion_vec) * i * 0.01
    #             alljnts[0] = pos1[0]
    #             alljnts[1] = pos1[1]
    #             temp_linear_conf.append(np.array(alljnts))
    #         linear_conf_list = temp_linear_conf
    #         # creat obj and jaw list
    #         linear_objpose_list = []
    #         linear_jawwidth_list = []
    #         for i, armjnts in enumerate(linear_conf_list):
    #             temp_objpose = copy.deepcopy(objpose_list2[-1])
    #             robot_s.fk(component_name="all", jnt_values=armjnts)
    #             eepos, eerot = robot_s.get_gl_tcp(manipulator_name="arm")
    #             # pos = temp_objpose[:3, 3] + rm.unit_vector(motion_vec) * i * 0.01
    #             linear_jawwidth_list.append(jawwidth_list2[-1])
    #             temp_objpose[:3, 3] = copy.deepcopy(eepos)
    #             linear_objpose_list.append(temp_objpose)
    #     else:
    #         print("--- 2 ----")
    #         temp_linear_conf = linear_conf_list.copy()
    #         tool.set_homomat(objpose_list2[-1])
    #         # gen linear_objpose_list
    #         objcm_copy = tool.copy()
    #         rel_obj_pos, rel_obj_rotmat = robot_s.hold(hnd_name=hnd_name, obj_cmodel=objcm_copy)
    #         robot_s.release(hnd_name=hnd_name, obj_cmodel=objcm_copy)
    #         linear_objpose_list = ppp_s.gen_object_motion(component_name="arm", conf_list=temp_linear_conf, obj_pos=rel_obj_pos,
    #                                                       obj_rotmat=rel_obj_rotmat, end_type='relative')
    #         # gen linear_objpose_list_box
    #         objcm_copy = target_obj.copy()
    #         rel_obj_pos, rel_obj_rotmat = robot_s.hold(hnd_name=hnd_name, obj_cmodel=objcm_copy)
    #         linear_objpose_list_box = ppp_s.gen_object_motion(component_name="arm", conf_list=temp_linear_conf,
    #                                                           obj_pos=rel_obj_pos, obj_rotmat=rel_obj_rotmat,
    #                                                           end_type='relative')
    #         robot_s.release(hnd_name=hnd_name, obj_cmodel=objcm_copy)
    #         linear_conf_list = robot_s.cvt_to_alljnts(conf_list_arm=temp_linear_conf, hnd_name=hnd_name)
    #         linear_jawwidth_list = []
    #         for i, armjnts in enumerate(linear_conf_list):
    #             linear_jawwidth_list.append(jawwidth_list2[-1])
    #
    #     ## plan rrt to the dist_pos
    #     path = rrtc_planner.plan(component_name="agv",
    #                                      start_conf=jnt_values_initial[:3],
    #                                      end_conf=dict_goal_pos,
    #                                      obstacle_list=[],
    #                                      other_robot_list=[],
    #                                      ext_dist=.05,
    #                                      max_n_iter=300,
    #                                      max_time=15.0,
    #                                      smoothing_n_iter=50,
    #                                      animation=False)
    #     print("path", path)
    #     if path is None:
    #         continue
    #     else:
    #         print("---succeeded rrt to dist pos---")
    #         # objpose_list_box for ppp
    #         arm_jaw = jnt_values_initial[3:11].tolist()
    #         objpose_list = []
    #         jawwidth_list = []
    #         objpose_list_box = []
    #         conf_list = []
    #         for i in path:
    #             agv = i.tolist()
    #             all = agv + arm_jaw
    #             conf_list.append(np.array(all))
    #             objpose_list.append(tool_initial.get_homomat())
    #             objpose_list_box.append(target_obj.get_homomat())
    #             jawwidth_list.append(robot_s.get_jawwidth(hand_name=hnd_name))
    #         # unite path (rrt + ppp)
    #         conf_list2 = robot_s.cvt_to_alljnts(conf_list_arm=conf_list2, hnd_name=hnd_name)
    #         conf_list += conf_list2
    #         objpose_list += objpose_list2
    #         jawwidth_list += jawwidth_list2
    #         objpose_list_box += objpose_list_box2
    #         # unite path (slide)
    #         conf_list += linear_conf_list
    #         objpose_list += linear_objpose_list
    #         jawwidth_list += linear_jawwidth_list
    #         objpose_list_box += linear_objpose_list_box
    #         print("---finish creating path---")
    #         break



## gen animation
robot_attached_list = []
object_attached_list = []
counter = [0]

def update(robot_s,
           hnd_name,
           object_box,
           target_obj,
           robot_path,
           jawwidth_path,
           obj_path,
           obj_path_box,
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
    robot_s.fk(component_name="all", joint_values=pose)
    robot_s.change_jaw_width(hnd_name, jawwidth_path[counter[0]])
    robot_meshmodel = robot_s.gen_mesh_model()
    robot_meshmodel.attach_to(base)
    robot_attached_list.append(robot_meshmodel)
    obj_pose = obj_path[counter[0]]
    obj_pose1 = obj_path_box[counter[0]]
    objb_copy = object_box.copy()
    objb_copy1 = target_obj.copy()
    objb_copy.set_homomat(obj_pose)
    objb_copy1.set_homomat(obj_pose1)
    objb_copy.attach_to(base)
    objb_copy1.attach_to(base)
    object_attached_list.append(objb_copy)
    object_attached_list.append(objb_copy1)
    counter[0] += 1
    return task.again
taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[robot_s,
                                 hnd_name,
                                 object_box,
                                 target_obj,
                                 conf_list,
                                 jawwidth_list,
                                 objpose_list,
                                 objpose_list_box,
                                 robot_attached_list,
                                 object_attached_list,
                                 counter],
                      appendTask=True)

base.run()