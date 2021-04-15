import os
import time
import math
import basis
import numpy as np
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import basis.robot_math as rm
if __name__ == '__main__':
    import copy
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.rpc.rviz_client as rv_client
    # # local code
    global_frame = gm.gen_frame()
    # define robot_s and robot_s anime info
    robot_s = xav.XArm7YunjiMobile()
    robot_meshmodel_parameters = [None,  # tcp_jntid
                                  None,  # tcp_loc_pos
                                  None,  # tcp_loc_rotmat
                                  False,  # toggle_tcpcs
                                  False,  # toggle_jntscs
                                  [0, .7, 0, .3],  # rgba
                                  'auto']  # name
    # define object and object anime info
    objfile = os.path.join(basis.__path__[0], 'objects', 'box.stl')
    obj = cm.CollisionModel(objfile)
    obj_parameters = [[0, 0, 1, 1]]  # rgba
    # obj_path = []
    # obj_path.append([np.array([.85, 0, .17]), np.eye(3)])  # [pos, rotmat]
    obj_path = [[np.array([.85, 0, .17]), np.eye(3)]]  # [pos, rotmat]
    print("obj_path =",obj_path)
    # a = np.array([0,0,0,0])
    # b = np.array(obj_path)
    # print(b.size)
    # obj_path = []
    # for i in range(10):
    #     obj_path.append([[np.array([.85, 0, .17 + i * 0.1]), np.eye(3)]])  # [pos, rotmat]
    obj.set_pos(np.array([1.2, 0, .27]))
    rotmat = rm.rotmat_from_axangle([1, 0, 0], -math.pi / 4.0)
    # obj.set_rotmat(np.array([[1,0,1],[0,1,0],[1,0,0]]))
    # # print(obj_path)
    # # obj.set_homomat(obj_path[0])
    # print("pos, rotmat",obj.get_pos(), obj.get_rotmat())
    obj.set_rgba([0, 1, 0, 1])
    # object list
    object_list = []
    obj_order = [0.018, 0.015, 0.018, 0.018, 0.018, 0.017, 0.015, 0.013]
    obj_number = [5,7,2,3,4,3,8,9]
    height = 0
    j = 0
    # for j, i in enumerate(obj_number):
    #     # object_list.append(os.path.join(basis.__path__[0], 'objects', 'obj'+str(i)+'.stl'))
    #     objfile = os.path.join(basis.__path__[0], 'objects', 'obj'+str(i)+'.stl')
    #     object_list.append(cm.CollisionModel(objfile))
    #     object_list[j].set_pos(np.array([1.5, 0, height + 0.1]))
    #     object_list[j].set_rgba([0.45098039215,0.30588235294,0.18823529411,.7])
    #     # object_list[j].set_rotmat(rotmat)
    #     height += obj_order[j]
    #
    # for object_temp in object_list:
    #     object, j = object_temp
    #     object.set_pos([0, 1, height])
    #     object.set_rgba([.5, .7, .3, 1])
    #     object.attach_to(base)
    #     height += j
    #     print(height)
    print(object_list)
    # remote code
    rvc = rv_client.RVizClient(host="localhost:182001")
    rvc.reset()
    rvc.load_common_definition(__file__, line_ids = range(1,8))
    rvc.change_campos_and_lookatpos(np.array([5,0,2]), np.array([0,0,.5]))
    # copy to remote
    rmt_global_frame = rvc.showmodel_to_remote(global_frame)
    # for i in range(len(object_list)):
    #     rmt_bunny = rvc.showmodel_to_remote(object_list[i])
    rmt_robot_s = rvc.copy_to_remote(robot_s)
    print("rbt",rmt_robot_s)
    rmt_obj = rvc.copy_to_remote(obj)
    print("obj",rmt_obj)
    # rvc.show_stationary_obj(rmt_obj)
    robot_component_name = 'all'
    a = np.array([0, 0, 0, 0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0,0])
    if not isinstance(a, np.ndarray) or a.size != 11:
        print("f!!")
    robot_s.fk(component_name=robot_component_name, jnt_values=np.array([0,0,0,0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0,0]))
    rrtc_planner = rrtc.RRTConnect(robot_s)
    # path = rrtc_planner.plan(start_conf=np.array([0,0,0,0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0,0]),
    #                          goal_conf=np.array([0,0,0.2,0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0,0]),
    #                          obstacle_list=object_list,
    #                          ext_dist=.1,
    #                          rand_rate=70,
    #                          maxtime=300,
    #                          component_name=robot_component_name)
    path = []
    for i in range(10):
        path.append(np.array([0,0,i*0.05,0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0,0]))
    print("path=",path)
    # robot_s.jaw_to(0.8)
    rmt_anime_robotinfo = rvc.add_anime_robot(rmt_robot_s=rmt_robot_s,
                                              loc_robot_component_name=robot_component_name,
                                              loc_robot_meshmodel_parameters=robot_meshmodel_parameters,
                                              loc_robot_motion_path=path)
    # print("rbt00000000",rmt_anime_robotinfo)
    # for i in obj_path:
    #     print("poooooooooo", i)
    #     m, n = i
    #     # t = np.array(i)
    #     print()
    # print(objfile)
    # print(rmt_obj)
    rmt_anime_objtinfo = rvc.add_anime_obj(rmt_obj=rmt_obj,
                      loc_obj=obj,
                      loc_obj_path=obj_path,
                      given_rmt_anime_objinfo_name=None)
    # print("obj11111111111",rmt_anime_objtinfo)
    # rmt_robot_meshmodel = rvc.add_stationary_robot(rmt_robot_s=rmt_robot_s,
    #                                                loc_robot_s=robot_s)
    time.sleep(1)
    # # draw sequence, problem: cannot work together with anime? (lost poses) -> cannot use the same remote instance
    # rmt_robot_mesh_list = []
    # newpath = copy.deepcopy(path)
    # rmt_robot_s2 = rvc.copy_to_remote(robot_s)
    # while True:
    #     for pose in newpath:
    #         robot_s.fk(component_name='arm', jnt_values=pose)
    #         # rmt_robot_mesh_list.append(rvc.showmodel_to_remote(robot_s.gen_meshmodel()))
    #         rmt_robot_mesh_list.append(rvc.add_stationary_robot(rmt_robot_s2, robot_s))
    #         time.sleep(.1)
    #     rvc.reset()
    #     rvc.load_common_definition(__file__, line_ids=range(1, 8))
    #     rvc.change_campos_and_lookatpos(np.array([5, 0, 2]), np.array([0, 0, .5]))
    #     time.sleep(.1)
    # rvc.delete_anime_robot(rmt_anime_robotinfo)
    # rvc.delete_stationary_robot(rmt_robot_meshmodel)
    # robot_s.fk(path[-1], component_name=robot_component_name)
    # rmt_robot_meshmodel = rvc.add_stationary_robot(rmt_robot_s='robot_s', loc_robot_s=robot_s)
    # obj.set_pos(obj.get_pos()+np.array([0,.1,0]))
    # obj.set_rgba([1,0,0,1])
    # rvc.update_remote(rmt_bunny, obj)