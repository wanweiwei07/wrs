import math
import time
import keyboard
import numpy as np
from wrs import basis as rm, robot_sim as rbs, modeling as gm
import wrs.visualization.panda.world as wd
# import robot_con.xarm_shuidi_grpc.xarm_shuidi_client as rbx

from direct.task.TaskManagerGlobal import taskMgr

base = wd.World(cam_pos=[3, 1, 1.5], lookat_pos=[0, 0, 0.5])
rbt_s = rbs.XArmShuidi()
# rbt_s.gen_meshmodel().attach_to(base)
# base.run()
# rbt_x.agv_move(agv_linear_speed=-.1, agv_angular_speed=.1, time_intervals=5)
onscreen = []
onscreen_agv = []
operation_count = 0
pre_pos = np.array([0, 0, math.radians(0)])
pre_angle = math.radians(0)

def agv_move(task):
    global onscreen, onscreen_agv, operation_count, pre_pos, pre_angle

    # agv_linear_speed = .002
    agv_linear_speed = 0
    agv_angular_speed = .5
    arm_linear_speed = .03
    arm_angular_speed = .1

    for item in onscreen:
        item.detach()
    onscreen.clear()
    for item in onscreen_agv:
        item.detach()
    onscreen_agv.clear()

    agv_pos = rbt_s.get_jnt_values("agv")
    agv_loc_rotmat = rm.rotmat_from_axangle(axis=[0, 0, 1], angle=-1*agv_pos[2])
    print(agv_pos)

    agv_direction = np.dot(np.array([1, 0, 0]), agv_loc_rotmat)
    # print(agv_direction)
    # onscreen_agv.append(mgm.gen_dasharrow(spos=np.array([agv_pos[0], agv_pos[1], 0]), epos=np.array([agv_pos[0], agv_pos[1], 0]))+np.array(agv_direction))
    onscreen_agv.append(gm.gen_arrow(epos=np.array([agv_direction[0], agv_direction[1], agv_direction[2]]), rgba=[1,0,1,1]))
    # onscreen_agv.append(mgm.gen_frame(pos=np.array([agv_pos[0], agv_pos[1], 0]), rotmat=agv_loc_rotmat, axis_length=3))
    onscreen_agv[-1].attach_to(base)
    # print(agv_pos)

    pressed_keys = {'w': keyboard.is_pressed('w'),
                    'a': keyboard.is_pressed('a'),
                    's': keyboard.is_pressed('s'),
                    'd': keyboard.is_pressed('d'),
                    'r': keyboard.is_pressed('r'),  # x+ global
                    't': keyboard.is_pressed('t'),  # x- global
                    'f': keyboard.is_pressed('f'),  # y+ global
                    'g': keyboard.is_pressed('g'),  # y- global
                    'v': keyboard.is_pressed('v'),  # z+ global
                    'b': keyboard.is_pressed('b'),  # z- gglobal
                    'y': keyboard.is_pressed('y'),  # r+ global
                    'u': keyboard.is_pressed('u'),  # r- global
                    'h': keyboard.is_pressed('h'),  # p+ global
                    'j': keyboard.is_pressed('j'),  # p- global
                    'n': keyboard.is_pressed('n'),  # yaw+ global
                    'm': keyboard.is_pressed('m'),  # yaw- global
                    'o': keyboard.is_pressed('o'),  # grippers open
                    'p': keyboard.is_pressed('p')}  # grippers close
    values_list = list(pressed_keys.values())
    if pressed_keys["w"] and pressed_keys["a"]:
        print("Invalid Operation!!")
        # rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
    elif pressed_keys["w"] and pressed_keys["d"]:
        print("Invalid Operation!!")
        # rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
    elif pressed_keys["s"] and pressed_keys["a"]:
        print("Invalid Operation!!")
        # rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
    elif pressed_keys["s"] and pressed_keys["d"]:
        print("Invalid Operation!!")
        # rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
    elif pressed_keys["w"] and sum(values_list) == 1:  # if key 'q' is pressed
        rbt_s.fk(component_name='agv', jnt_values=np.array(pre_pos + [agv_linear_speed, 0, math.radians(0.5)]))
        pre_pos = np.array(pre_pos + [agv_linear_speed, 0, math.radians(0.5)])
        # rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=0, time_interval=.5)
    elif pressed_keys["s"] and sum(values_list) == 1:  # if key 'q' is pressed
        rbt_s.fk(component_name='agv', jnt_values=np.array(pre_pos + [-1*agv_linear_speed, 0, math.radians(-0.5)]))
        pre_pos = np.array(pre_pos + [-1*agv_linear_speed, 0, math.radians(-0.5)])
        # rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=0, time_interval=.5)
    elif pressed_keys["a"] and sum(values_list) == 1:  # if key 'q' is pressed
        rbt_s.fk(component_name='agv', jnt_values=np.array(pre_pos + [0, agv_linear_speed, math.radians(0)]))
        pre_pos = np.array(pre_pos + [0, agv_linear_speed, math.radians(0)])
        # rbt_x.agv_move(linear_speed=0, angular_speed=agv_angular_speed, time_interval=.5)
    elif pressed_keys["d"] and sum(values_list) == 1:  # if key 'q' is pressed
        rbt_s.fk(component_name='agv', jnt_values=np.array(pre_pos + [0, -1*agv_linear_speed, math.radians(0)]))
        pre_pos = np.array(pre_pos + [0, -1*agv_linear_speed, math.radians(0)])
        # rbt_x.agv_move(linear_speed=0, angular_speed=-agv_angular_speed, time_interval=.5)
    elif pressed_keys["o"] and sum(values_list) == 1:  # if key 'q' is pressed
        print("Invalid Operation!!")
        # rbt_x.arm_jaw_to(ee_values=100)
    elif pressed_keys["p"] and sum(values_list) == 1:  # if key 'q' is pressed
        print("Invalid Operation!!")
        # rbt_x.arm_jaw_to(ee_values=0)
    elif any(pressed_keys[item] for item in ['r', 't', 'f', 'g', 'v', 'b', 'y', 'u', 'h', 'j', 'n', 'm']) and\
            sum(values_list) == 1: # global
        tic = time.time()
        current_jnt_values = rbt_s.get_jnt_values()
        current_arm_tcp_pos, current_arm_tcp_rotmat = rbt_s.get_gl_tcp()
        rel_pos = np.zeros(3)
        rel_rotmat = np.eye(3)
        if pressed_keys['r']:
            rel_pos = np.array([arm_linear_speed * .5, 0, 0])
        elif pressed_keys['t']:
            rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
        elif pressed_keys['f']:
            rel_pos = np.array([0, arm_linear_speed * .5, 0])
        elif pressed_keys['g']:
            rel_pos = np.array([0, -arm_linear_speed * .5, 0])
        elif pressed_keys['v']:
            rel_pos = np.array([0, 0, arm_linear_speed * .5])
        elif pressed_keys['b']:
            rel_pos = np.array([0, 0, -arm_linear_speed * .5])
        elif pressed_keys['y']:
            rel_rotmat = rm.rotmat_from_euler(arm_angular_speed*.5, 0, 0)
        elif pressed_keys['u']:
            rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed*.5, 0, 0)
        elif pressed_keys['h']:
            rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed*.5, 0)
        elif pressed_keys['j']:
            rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
        elif pressed_keys['n']:
            rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed*.5)
        elif pressed_keys['m']:
            rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed*.5)
        new_arm_tcp_pos = current_arm_tcp_pos+rel_pos
        new_arm_tcp_rotmat = rel_rotmat.dot(current_arm_tcp_rotmat)
        last_jnt_values = rbt_s.get_jnt_values()
        new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos, tgt_rotmat=new_arm_tcp_rotmat, seed_jnt_values=current_jnt_values)
        # if new_jnt_values is None:
        #     continue
        rbt_s.fk(jnt_values=new_jnt_values)
        toc = time.time()
        start_frame_id = math.ceil((toc - tic) / .01)
        # rbt_x.arm_move_jspace_path([last_jnt_values, new_jnt_values], time_interval=.1, start_frame_id=start_frame_id)

    onscreen.append(rbt_s.gen_meshmodel())
    onscreen[-1].attach_to(base)
    # print(pre_pos)
    # print(onscreen)

    operation_count += 1
    # time.sleep(1/30)
    return task.cont

if __name__ == '__main__':
    gm.gen_frame(axis_length=3, axis_radius=0.01).attach_to(base)
    # threading.Thread(target=agv_move, args=()).start()
    taskMgr.doMethodLater(1/60, agv_move, "agv_move",
                          extraArgs=None,
                          appendTask=True)
    base.run()