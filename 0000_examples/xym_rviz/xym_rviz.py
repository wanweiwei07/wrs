import os
import time
import math
import basis
import numpy as np
from basis import robotmath as rm
import visualization.panda.world as wd
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import motion.probabilistic.rrt_connect as rrtc
import visualization.panda.rpc.rviz_client as rv_client


rvc = rv_client.RVizClient(host="192.168.1.111:182001")
rvc.change_campos_and_lookatpos(np.array([3,0,2]), np.array([0,0,.5]))
# run common_definition at remote end
rvc.load_common_definition('common_definition.py')
# run common definition at local end
exec(rvc.common_definition, globals())

global obj
global robot_instance
global robot_jlc_name
global robot_meshmodel_parameters

# # local code
obj.set_pos(np.array([.85, 0, .17]))
obj.set_rgba([.5, .7, .3, 1])
rvc.clear_obj_render_info_list()
# rvc.add_obj_render_info(obj)
jlc_name = 'arm'
robot_instance.fk(np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]), jlc_name=jlc_name)
rrtc_planner = rrtc.RRTConnect(robot_instance)
path = rrtc_planner.plan(start_conf=np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]),
                         goal_conf=np.array([math.pi / 3, math.pi * 1 / 3, 0, math.pi / 2, 0, math.pi / 6, 0]),
                         obstacle_list=[obj],
                         ext_dist=.1,
                         rand_rate=70,
                         maxtime=300,
                         jlc_name=jlc_name)
rvc.clear_robot_render_info_list()
rvc.add_robot_render_info(robot_jlc_name, robot_meshmodel_parameters, path)
# rvc.run_code("start_rendering_task(robot_render_info_list, obj_render_info_list)")
#
# # base = wd.World(campos=[1, 1, 1], lookatpos=[0, 0, 0])
# # jlc_name='arm'
# # robot_instance = xav.XArm7YunjiMobile()
# # robot_instance.fk(np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]), jlc_name=jlc_name)
# # robot_meshmodel = robot_instance.gen_meshmodel()
# # robot_meshmodel.attach_to(base)
# # path_counter = 0
# # def rviz_task(robot_instance_list, robot_meshmodel_list, path_counter_list, path_list, jlc_name_list, task):
# #     for i in range(len(robot_instance_list)):
# #         robot_instance_list[i].fk(path_list[i][path_counter_list[i]], jlc_name=jlc_name_list[i])
# #         robot_meshmodel_list[i].detach()
# #         robot_meshmodel_list[i] = robot_instance_list[i].gen_meshmodel()
# #         robot_meshmodel_list[i].attach_to(base)
# #         path_counter_list[i] += 1
# #         if path_counter_list[i] >= len(path_list[i]):
# #             path_counter_list[i] = 0
# #     return task.cont
# # taskMgr.add(rviz_task, extraArgs=[[robot_instance], [robot_meshmodel], [path_counter], [path], [jlc_name]], appendTask=True)
# # base.run()
#
# create_robot_counter = "robot_path_counter = 0\n"
# rvc.run_code(create_robot_counter)
#
# create_robot_path = "robot_path = ["
# for pose in path:
#     create_robot_path += "np.array(%s)," % np.array2string(pose, separator=',')
# create_robot_path = create_robot_path[:-1] + "]\n"
# rvc.run_code(create_robot_path)
#
# create_robot_meshmodel_parameter = ("robot_meshmodel_parameter = [" +
#                                     "None, " +  # tcp_jntid
#                                     "None, " +  # tcp_loc_pos
#                                     "None, " +  # tcp_loc_rotmat
#                                     "False, " +  # toggle_tcpcs
#                                     "False, " +  # toggle_jntscs
#                                     "[0,.3,0,0.5], " +  # rgba
#                                     "'auto']\n")  # name
# rvc.run_code(create_robot_meshmodel_parameter)
#
# create_obj = "obj = None\n" + \
#              "obj_parameter = None\n" + \
#              "obj_path = None\n" + \
#              "obj_path_counter = None\n"
# rvc.run_code(create_obj)
#
# create_showloop_code = \
# '''
# def rviz_task(robot_instance_list,
#               robot_meshmodel_list,
#               robot_meshmodel_parameter_list,
#               robot_path_list,
#               robot_path_counter_list,
#               robot_jlc_name_list,
#               obj_list,
#               obj_parameter_list,
#               obj_path_list,
#               obj_path_counter_list,
#               task):
#     for i in range(len(robot_instance_list)):
#         if robot_instance_list[0] is None:
#             break
#         robot_meshmodel_list[i].detach()
#         robot_instance_list[i].fk(robot_path_list[i][robot_path_counter_list[i]], jlc_name=robot_jlc_name_list[i])
#         robot_meshmodel_list[i] = robot_instance_list[i].gen_meshmodel(tcp_jntid=robot_meshmodel_parameter_list[i][0],
#                                                                        tcp_loc_pos=robot_meshmodel_parameter_list[i][1],
#                                                                        tcp_loc_rotmat=robot_meshmodel_parameter_list[i][2],
#                                                                        toggle_tcpcs=robot_meshmodel_parameter_list[i][3],
#                                                                        toggle_jntscs=robot_meshmodel_parameter_list[i][4],
#                                                                        rgba=robot_meshmodel_parameter_list[i][5],
#                                                                        name=robot_meshmodel_parameter_list[i][6],)
#         robot_meshmodel_list[i].attach_to(base)
#         robot_path_counter_list[i] += 1
#         if robot_path_counter_list[i] >= len(robot_path_list[i]):
#             robot_path_counter_list[i] = 0
#     for i in range(len(obj_list)):
#         if obj_list[0] is None:
#             break
#         obj_list[i].detach()
#         obj_list[i].set_pos(obj_path_list[i][obj_path_counter_list[i]][0])
#         obj_list[i].set_mat(obj_path_list[i][obj_path_counter_list[i]][1])
#         obj_list[i].set_rgba(obj_parameter_list[i])
#         obj_list[i].attach_to(base)
#         obj_path_counter_list[i] += 1
#         if obj_path_counter_list[i] >= len(obj_path_list[i]):
#             obj_path_counter_list[i] = 0
#     return task.cont
# taskMgr.doMethodLater(0.1, rviz_task, "rviz_task",
#                       extraArgs=[[robot_instance],
#                                  [robot_meshmodel],
#                                  [robot_meshmodel_parameter],
#                                  [robot_path],
#                                  [robot_path_counter],
#                                  [robot_jlc_name],
#                                  [obj],
#                                  [obj_parameter],
#                                  [obj_path],
#                                  [obj_path_counter]],
#                       appendTask=True)
# '''
# rvc.run_code(create_showloop_code)
# time.sleep(3)
# rvc.clear_task()
# # while True:
# #     for pose in path:
# #         update_robot = \
# # '''
# # robot_meshmodel.detach()
# # robot_instance.fk(np.array(%s), jlc_name=jlc_name)
# # robot_meshmodel = robot_instance.gen_meshmodel()
# # robot_meshmodel.attach_to(base)
# # ''' % np.array2string(pose, separator=',')
# #         rvc.run_code(update_robot)
# #         # robot_meshmodel = robot_instance.gen_meshmodel()
# #         # robot_meshmodel.attach_to(base)
# #         # # robot_meshmodel.show_cdprimit()
# #         # robot_instance.gen_stickmodel().attach_to(base)
# #         time.sleep(.04)
#
# # print(path)
# # for pose in path:
# #     print(pose)
# #     robot_instance.fk(pose, jlc_name=jlc_name)
# #     robot_meshmodel = robot_instance.gen_meshmodel()
# #     robot_meshmodel.attach_to(base)
# #     # robot_meshmodel.show_cdprimit()
# #     robot_instance.gen_stickmodel().attach_to(base)
# # hold
# # robot_instance.hold(object, jawwidth=.05)
# # robot_instance.fk(np.array([0, 0, 0, math.pi/6, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
# # robot_meshmodel = robot_instance.gen_meshmodel()
# # robot_meshmodel.attach_to(base)
# # robot_instance.show_cdprimit()
# # tic = time.time()
# # result = robot_instance.is_collided() # problematic
# # toc = time.time()
# # print(result, toc - tic)
# # base.run()
# # release
# # robot_instance.release(object, jawwidth=.082)
# # robot_instance.fk(np.array([0, 0, 0, math.pi/3, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, math.pi/6]))
# # robot_meshmodel = robot_instance.gen_meshmodel()
# # robot_meshmodel.attach_to(base)
# # robot_meshmodel.show_cdprimit()
# # tic = time.time()
# # result = robot_instance.is_collided()
# # toc = time.time()
# # print(result, toc - tic)
#
# # copy
# # robot_instance2 = robot_instance.copy()
# # robot_instance2.move_to(pos=np.array([.5,0,0]), rotmat=rm.rotmat_from_axangle([0,0,1], math.pi/6))
# # objcm_list = robot_instance2.get_hold_objlist()
# # robot_instance2.release(objcm_list[-1], jawwidth=.082)
# # robot_meshmodel = robot_instance2.gen_meshmodel()
# # robot_meshmodel.attach_to(base)
# # robot_instance2.show_cdprimit()
#
# # base.run()
