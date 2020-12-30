import os
import math
import basis
import numpy as np
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav
import visualization.panda.rpc.render_info as rdi


def create_robot_render_info(robot_instance,
                             robot_jlc_name,
                             robot_meshmodel_parameters,
                             robot_path):
    robot_render_info = rdi.RobotInfo()
    robot_render_info.robot_instance = robot_instance
    robot_render_info.robot_jlc_name = robot_jlc_name
    robot_render_info.robot_meshmodel = robot_instance.gen_meshmodel(robot_meshmodel_parameters)
    robot_render_info.robot_meshmodel_parameters = robot_meshmodel_parameters
    robot_render_info.robot_path = robot_path
    robot_render_info.robot_path_counter = 0
    return robot_render_info

def create_obj_render_info(obj,
                           obj_parameters=None,
                           obj_path = None):
    obj_render_info = rdi.ObjInfo()
    obj_render_info.obj = obj
    if obj_parameters is None:
        obj_render_info.obj_parameters = obj.get_rgba()
    else:
        obj_render_info.obj_parameters = obj_parameters
    if obj_path is None:
        obj_render_info.obj_path = [[obj.get_pos(), obj.get_rotmat()]]
    else:
        obj_render_info.obj_path = obj_path
    obj_render_info.obj_path_counter = 0
    return obj_render_info


def start_rendering_task(robot_render_info_list, obj_render_info_list):
    def rviz_task(robot_render_info_list,
                  obj_render_info_list,
                  task):
        for i in range(len(robot_render_info_list)):
            robot_instance = robot_render_info_list[i].robot_instance
            robot_jlc_name = robot_render_info_list[i].robot_jlc_name
            robot_meshmodel = robot_render_info_list[i].robot_meshmodel
            robot_meshmodel_parameter = robot_render_info_list[i].robot_meshmodel_parameters
            robot_path = robot_render_info_list[i].robot_path
            robot_path_counter = robot_render_info_list[i].robot_path_counter
            robot_meshmodel.detach()
            robot_instance.fk(robot_path[robot_path_counter], jlc_name=robot_jlc_name)
            robot_render_info_list[i].robot_meshmodel = robot_instance.gen_meshmodel(tcp_jntid=robot_meshmodel_parameter[0],
                                                                                     tcp_loc_pos=robot_meshmodel_parameter[1],
                                                                                     tcp_loc_rotmat=robot_meshmodel_parameter[2],
                                                                                     toggle_tcpcs=robot_meshmodel_parameter[3],
                                                                                     toggle_jntscs=robot_meshmodel_parameter[4],
                                                                                     rgba=robot_meshmodel_parameter[5],
                                                                                     name=robot_meshmodel_parameter[6], )
            robot_render_info_list[i].robot_meshmodel.attach_to(base)
            robot_render_info_list[i].robot_path_counter += 1
            if robot_render_info_list[i].robot_path_counter >= len(robot_path):
                robot_render_info_list[i].robot_path_counter = 0
        for i in range(len(obj_render_info_list)):
            obj = obj_render_info_list[i].obj
            obj_parameters = obj_render_info_list[i].obj_parameters
            obj_path = obj_render_info_list[i].obj_path
            obj_path_counter = obj_render_info_list[i].obj_path_counter
            obj.detach()
            obj.set_pos(obj_path[obj_path_counter][0])
            obj.set_rotmat(obj_path[obj_path_counter][1])
            obj.set_rgba(obj_parameters[0])
            obj.attach_to(base)
            obj_render_info_list[i].obj_path_counter += 1
            if obj_render_info_list[i].obj_path_counter >= len(obj_path):
                obj_render_info_list[i].obj_path_counter = 0
        return task.cont

    taskMgr.doMethodLater(0.3, rviz_task, "rviz_task",
                          extraArgs=[robot_render_info_list,
                                     obj_render_info_list],
                          appendTask=True)


robot_render_info_list = []
obj_render_info_list = []
# define global frame
global_frame = gm.gen_frame()
# define robot and robot render info
robot_instance = xav.XArm7YunjiMobile()
robot_jlc_name = 'arm'
robot_meshmodel_parameters = [None,  # tcp_jntid
                              None,  # tcp_loc_pos
                              None,  # tcp_loc_rotmat
                              False,  # toggle_tcpcs
                              False,  # toggle_jntscs
                              [0, .7, 0, .3],  # rgba
                              'auto']  # name
robot_path = [np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0])]
robot_render_info = create_robot_render_info(robot_instance,
                                             robot_jlc_name,
                                             robot_meshmodel_parameters,
                                             robot_path)
robot_render_info_list.append(robot_render_info)
# define object and object render info
objfile = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
obj = cm.CollisionModel(objfile)
obj_parameters = [[.3, .2, .1, 1]]  # rgba
obj_path = [[np.array([.85, 0, .17]), np.eye(3)]]  # [pos, rotmat]
obj_render_info = create_obj_render_info(obj, obj_parameters, obj_path)
obj_render_info_list.append(obj_render_info)
