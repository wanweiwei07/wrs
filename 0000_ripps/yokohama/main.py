import copy
import math
import os.path
import time
import cv2

import numpy as np
import torch

from wrs import basis as rm, robot_sim as cbtr, motion as rrtc, modeling as gm
import cobotta_x as cbtx
import env_bulider_RIKEN as eb
from realsensecrop import RealSenseD405Crop
import model_loader as model
import wrs.visualization.panda.world as wd
import serial

ZERO_ERROR = 0.035 - 0.0048
temp_pose = np.array([0.2, 0.2, 0.2, math.pi / 2, 0, 2, -1])


def get_experiment_conf(requirement_doc, plant_id, chemical_temp):
    _, chemical_id, pos_id = [row for row in requirement_doc if row[0] == str(plant_id)]
    tip_change_toggle = False
    if chemical_id != 0 and chemical_id != chemical_temp:
        tip_change_toggle = True
        chemical_temp = copy.deepcopy(chemical_id)
    return chemical_id, pos_id, chemical_temp, tip_change_toggle


experiment_pos = {"N": "N/A",
                  "C": "Center",
                  "L": "Leaf",
                  "LL": "TwoLeaves"}


def get_start_plant_id():
    l_time = time.localtime()
    father_path = f"./capture/{l_time[0]}_{l_time[1]}_{l_time[2]}/"
    if not os.path.exists(father_path + "experiment_info.txt"):
        return 2, []
    else:
        exp_info = np.loadtxt(father_path + "experiment_info.txt", dtype=str)
        if len(exp_info.shape) < 2:
            exp_info = np.array([exp_info])
        return int(exp_info.T[1][-1][:-1]) + 1, exp_info


def save_exp_info(plant_info_list, experiment_info_document, img_origin, img_mask):
    plant_id, exp_target, chemical_id = plant_info_list
    cv2.imwrite(father_path + f"plant_{plant_id}_{exp_target}_chemical_{chemical_id}.png",
                plant_mask)
    cv2.imwrite(father_path + f"plant_{plant_id}_origin.png", img_origin)
    cv2.imwrite(father_path + f"plant_{plant_id}_mask.png", img_mask)
    experiment_info = np.array(
        [[f"plant_id: {plant_id}, ethanol: {chemical_id}, position: {exp_target}"]])
    experiment_info_document = np.append(experiment_info_document, experiment_info, axis=0)
    np.savetxt(father_path + "experiment_info.txt", experiment_info_document, fmt='%s')
    return experiment_info_document


class MotionPlannerRT():
    def __init__(self, robot_s, robot_x, obstacle_list=[], component_name="arm"):
        self.robot_s = robot_s
        self.robot_x = robot_x
        self.obstacle_list = obstacle_list
        self.component_name = component_name
        self.rrtc_planner = rrtc.RRTConnect(self.robot_s)
        self.create_angle_list()

    def get_rbt_pose_from_pipette(self, pipette_gl_pos, pipette_gl_angle=0, dist=0.007):
        tcp_gl_mat = self.get_tcp_gl_mat_from_pipette(pipette_gl_pos, pipette_gl_angle, dist)
        rbt_tcp_pos = np.array([0, 4.7, 10]) / 1000
        rbt_tcp_mat = rm.homomat_from_posrot(rbt_tcp_pos, np.eye(3))
        rbt_gl_mat = np.dot(tcp_gl_mat, rbt_tcp_mat)
        rot_euler = rm.rotmat_to_euler(rbt_gl_mat[:3, :3])
        return np.append(np.append(rbt_gl_mat[:3, 3], rot_euler), 5)

    def get_tcp_gl_mat_from_pipette(self, pipette_gl_pos, pipette_gl_angle=0, dist=0.007):
        pipette_gl_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(pipette_gl_angle))
        pipette_gl_mat = rm.homomat_from_posrot(pipette_gl_pos, pipette_gl_rot)
        pipette_tcp_pos = np.array([-0.008, -0.15485, 0.01075]) + np.array(
            [0.0015, -dist, -0.0058])  # [y,z,x] in global
        pipette_tcp_rot = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                 rm.rotmat_from_axangle(np.array([0, 1, 0]), -math.pi / 2))
        pipette_tcp_mat = rm.homomat_from_posrot(pipette_tcp_pos, pipette_tcp_rot)
        tcp_gl_mat = np.dot(pipette_gl_mat, np.linalg.inv(pipette_tcp_mat))
        return tcp_gl_mat

    def create_angle_list(self):
        angle_half = np.array(range(0, 185, 5))
        angle_list = np.array([0] * 73)
        angle_list[::2] = angle_half
        angle_list[1:][::2] = angle_half[1:] * -1
        # angle_list = angle_list - np.ones(len(angle_list)) * 90
        self.angle_list = list(angle_list)

    def move_planner(self, tgt_pos, direct_pose=False, is_plant=False):
        dist = 0.06
        if is_plant:
            dist = 0
        current_jnts = self.robot_x.get_jnt_values()

        for angle in self.angle_list:
            if direct_pose:
                tgt_pose = np.array([0, 0, dist, np.pi / 2, 0, np.pi / 2, 5])
                tgt_pose[:3] += tgt_pos[:3]
                tgt_pose[5] += np.radians(angle)
            else:
                sample_mat = self.get_tcp_gl_mat_from_pipette(tgt_pos, angle, dist)
                gm.gen_frame(sample_mat[:3, 3], sample_mat[:3, :3]).attach_to(base)
                tgt_pose = self.get_rbt_pose_from_pipette(tgt_pos, angle, dist)
            try:
                tgt_jnts = self.robot_x.P2J(tgt_pose)
                if not self.robot_s.are_jnts_in_ranges("arm", tgt_jnts):
                    print("out of range")
                    continue
            except:
                # print(tgt_pose)
                # mgm.gen_frame(tgt_pose[:3]).attach_to(base)
                self.robot_x.clear_error()
                continue
            # print(f"mat:{sample_mat}")

            if tgt_jnts is not None:
                return tgt_jnts

            #     print(tgt_jnts)
            #     self.robot_s.fk(jnt_values=tgt_jnts)
            #     if self.robot_s.is_collided():
            #         print("robot collision")
            #         continue
            #     # print(tgt_jnts)
            #     path = self.rrtc_planner.plan(component_name=self.component_name,
            #                                   start_conf=current_jnts,
            #                                   end_conf=tgt_jnts,
            #                                   ext_dist=0.2,
            #                                   max_n_iter=1000,
            #                                   obstacle_list=self.obstacle_list,
            #                                   smoothing_n_iter=100)
            #     # angle_output.append(angle)
            #     # robot_s.gen_meshmodel().attach_to(base)
            #     # base.run()
            #     return path

        print("No result")
        return None


def move_to_new_pose(pose, speed=100):
    pose, times = robot_x.null_space_search(pose)
    if pose is not None:
        robot_x.move_pose(pose, speed=speed)
        return times
    else:
        raise Exception("No solution!")


def attach_tip(tip_id, gl_err):
    '''
    pick up new tip
    '''
    total_err = copy.deepcopy(gl_err)
    tgt_pos = env.tip_rack._hole_pos_list[tip_id - 1]
    tgt_pos[:2] -= total_err  # closed-loop correction
    tgt_pos[2] = 0.2025
    # print("tgt_pos:", tgt_pos)
    move_result = False
    # base.run()

    jnts_tip = mplan.move_planner(tgt_pos, direct_pose=True)
    robot_x.move_jnts(jnts_tip)
    current_pose = robot_x.get_pose_values()
    # print("current_pose:", current_pose)
    current_pose[2] -= 0.052
    rotation_times = move_to_new_pose(current_pose)
    # print(rotation_times)

    '''
    adjust position
    '''
    err_list = [np.zeros(3)]
    step = 0
    pre = 0
    pre_pre = 0
    pre_pre_pre = 0

    while 1:
        if step > 10:
            break
        current_pose = robot_x.get_pose_values()
        time.sleep(0.3)
        img_o,img_1,img_2 =rs_pipeline.get_learning_feature()
        with torch.no_grad():
            direct_trans_tip, _, _ = model_vit.get_score(img_2)
        score = direct_trans_tip
        # print(score)

        if score == 0:
            break
        else:
            rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), current_pose[5] - math.pi)
            xy_array = rm.gen_2d_isosceles_verts(nlevel=4, edge_length=0.0009, nedges=6)
            print("judge:", pre_pre, pre, score)
            if score == pre_pre:
                print("repeated")
                xy_pose = xy_array[score] / 2
            elif min(score, pre, pre_pre) > 0 and max(score, pre, pre_pre) < 7:
                xy_pose = xy_array[score] / 2
            else:
                xy_pose = xy_array[score]
            pos_err = rotmat.dot(np.append(xy_pose, 0))
            total_err += pos_err[:2]
            current_pose[:2] -= pos_err[:2]
            times = move_to_new_pose(current_pose)
            print(times)
            rotation_times += times
            step += 1
            pre_pre_pre = pre_pre
            pre_pre = pre
            pre = score
            time.sleep(0.2)
    # print(err_list)
    # print(np.cumsum(err_list, axis=1))
    if step > 10:
        rise_pose = robot_x.get_pose_values()
        rise_pose[2] += 0.05
        move_to_new_pose(rise_pose)
        rise_pose[2] += 0.05
        move_to_new_pose(rise_pose)
        return gl_err, False

    '''
    insert tip
    '''
    # depth_adjust = abs((tip_id % 24) - 12) / 15000
    depth_base = 0.0225
    current_pose = robot_x.get_pose_values()
    insert_pose = copy.deepcopy(current_pose)
    # insert_pose[2] -= depth_base + depth_adjust
    insert_pose[2] -= depth_base
    move_to_new_pose(insert_pose)

    # insert_tip()
    insert_pose[2] -= 0.0025
    move_to_new_pose(insert_pose, 30)

    rise_pose = copy.deepcopy(current_pose)
    # rise_pose[2] -= 0.0065
    # # arise_tip(rise_pose)
    # move_to_new_pose(rise_pose, 20)

    # rise_pose[2] += 0.0065
    move_to_new_pose(rise_pose, 75)
    rise_pose[2] += 0.02
    move_to_new_pose(rise_pose)
    rise_pose[2] += 0.03
    move_to_new_pose(rise_pose)
    rise_pose[2] += 0.03
    move_to_new_pose(rise_pose)

    return total_err, True


def liquid_dispensing(chemical_id, plant_pose_gl, plant_center_pixel):
    '''
    get chemical
    '''
    move_result = False
    chemical_pos = env.deep_plate._hole_pos_list[chemical_id]
    chemical_pos[2] = 0.2065

    jnts_chemical = mplan.move_planner(chemical_pos, direct_pose=True)
    robot_x.move_jnts(jnts_chemical)
    current_pose = robot_x.get_pose_values()
    time.sleep(0.1)
    current_pose[2] -= 0.01
    robot_x.move_pose(temp_pose)
    robot_x.move_pose(current_pose)

    '''
    insert chemical
    '''
    # print("insert chemical")
    current_pose = robot_x.get_pose_values()
    time.sleep(0.1)
    insert_pose = copy.deepcopy(current_pose)
    robot_x.open_gripper(speed=90)
    insert_pose[2] -= 0.029
    move_to_new_pose(insert_pose, 50)
    insert_pose[2] -= 0.025
    move_to_new_pose(insert_pose, 50)
    robot_x.defult_gripper(speed=50)
    insert_pose[2] += 0.025
    move_to_new_pose(insert_pose, 50)
    move_to_new_pose(current_pose, 75)
    current_pose[2] += 0.025
    move_to_new_pose(current_pose, 75)

    '''
    water plant
    '''
    # print("water plant")
    plant_pose_rbt = copy.deepcopy(plant_pose_gl)
    center_coor = plant_center_pixel - np.array([150, 170])
    rbt_coor = center_coor * 70 / 260
    plant_pose_rbt[0] = plant_pose_rbt[0] + rbt_coor[0] / 1000
    plant_pose_rbt[1] = plant_pose_rbt[1] - rbt_coor[1] / 1000
    robot_x.move_pose(temp_pose)
    robot_x.move_pose(plant_pose_rbt)
    move_result = False

    jnts_plant = mplan.move_planner(plant_pose_rbt, direct_pose=True, is_plant=True)
    robot_x.move_jnts(jnts_plant)
    current_pose = robot_x.get_pose_values()
    # time.sleep(0.1)
    current_pose[2] -= 0.006
    move_to_new_pose(current_pose)
    robot_x.open_gripper(speed=70)
    robot_x.defult_gripper()
    robot_x.open_gripper(speed=100)
    robot_x.defult_gripper()
    current_pose = robot_x.get_pose_values()
    time.sleep(0.1)
    current_pose[2] += 0.05
    move_to_new_pose(current_pose)


def dispose_tip(tip_id):
    print("throw away tip")
    eject_jnt_values = eject_jnt_values_list[tip_id % 4]
    current_pose = robot_x.get_pose_values()
    current_pose[1] += 0.04
    current_pose[2] += 0.03
    move_to_new_pose(current_pose)
    robot_x.move_jnts(eject_jnt_values)
    current_pose = robot_x.get_pose_values()
    time.sleep(0.1)
    current_pose[2] -= 0.05
    move_to_new_pose(current_pose)
    robot_x.close_gripper()
    robot_x.defult_gripper()
    current_pose = robot_x.get_pose_values()
    time.sleep(0.1)
    current_pose[2] += 0.06
    move_to_new_pose(current_pose)


def is_inside_range(jnt_values):
    for i in range(6):
        if jnt_values[i] < robot_s.arm.jlc.jnts[i + 1]['motion_range'][0] or jnt_values[i] > \
                robot_s.arm.jlc.jnts[i + 1]['motion_range'][1]:
            # print(jnt_values[i], robot_s.arm.jlc.joints[i]['motion_range'])
            print(f"{i} out of range")
            # robot_s.fk(jnt_values=jnt_values)
            # robot_s.gen_meshmodel().attach_to(base)
            # base.run()
            return False
    return True


def detect_cts():
    cts_now = ser.cts
    while True:
        cts_pre = cts_now
        cts_now = ser.cts
        time.sleep(.1)
        # print(cts_pre,cts_now)
        if (not cts_pre) and cts_now:
            return


def wait_for_start():
    while True:
        local_time = time.localtime()
        print(local_time, f"hour:{local_time[3]}")
        # if local_time[3] > 9:
        img_o, img_plant = rs_pipeline.get_plant_feature()
        marker_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_plant, marker_dict)
        print(ids)
        if ids is not None:
            return
        detect_cts()

if __name__ == '__main__':
    '''
    set up    
    '''
    base = wd.World(cam_pos=[0.2445, 0.07175, 0.67275], lookat_pos=[0.2445, 0.07175, 0])
    gm.gen_frame().attach_to(base)

    component_name = "arm"
    manipulator_name = 'hnd',
    robot_s = cbtr.CobottaRIPPS()
    robot_x = cbtx.CobottaX()
    ser = serial.Serial(port="COM1", rtscts=True)
    print("connected")

    env = eb.Env(robot_s)
    mplan = MotionPlannerRT(robot_s, robot_x, obstacle_list=[env.tip_rack, env.deep_plate, env.frame_bottom])
    rrtc_planner = rrtc.RRTConnect(robot_s)

    eject_jnt_values1 = np.array([1.37435462, 0.98535585, 0.915062, 1.71130978, -1.23317083, -0.93993529])
    eject_jnt_values2 = np.array([1.21330229, 1.04185468, 0.84606752, 1.61686219, -1.11360179, -0.90834774])
    eject_jnt_values3 = np.array([1.41503157, 0.67831314, 1.41330569, 1.45821765, -1.41208026, -0.45279814])
    eject_jnt_values4 = np.array([1.16644694, 0.883306, 1.07903103, 1.47912526, -1.41216692, -0.64314669])
    eject_jnt_values_list = [eject_jnt_values1, eject_jnt_values2, eject_jnt_values3, eject_jnt_values4]

    plant_pose = np.array([0.2260, 0.2405, 0.2045, np.pi / 2, 0, np.pi / 2, 5])

    model_vit = model.TransformerModel("model/vit_yh_last", (120, 120), 10, 61)
    leaf_model = model.MaskRCNNModel("model/mask_rcnn_model")

    rs_pipeline = RealSenseD405Crop()
    rs_pipeline.get_learning_feature()
    time.sleep(0.5)
    rs_pipeline.get_learning_feature()

    local_time = time.localtime()
    print(local_time)

    tip_id = 0
    total_num = 0
    total_err = np.zeros(2)

    """
    reset position
    """
    detect_pose = copy.deepcopy(plant_pose)
    detect_pose[0] -= 0.025
    robot_x.move_pose(temp_pose)
    robot_x.move_pose(detect_pose)
    # robot_x.close_gripper()
    robot_x.defult_gripper()
    # input()

    detect_cts()
    father_path = f"./capture/{local_time[0]}_{local_time[1]}_{local_time[2]}/"
    plant_id, experiment_info_document = get_start_plant_id()
    print(f"start plant id: {plant_id}")

    file_path = "requirement.txt"
    requirement_doc = np.loadtxt(file_path, dtype=str)[1:]
    chemical_temp = 0
    is_tip_attached = False

    if plant_id == 2:
        wait_for_start()
        detect_cts()
        print("-" * 50)
        local_time = time.localtime()
        print(f"date:{local_time[0]}_{local_time[1]}_{local_time[2]}")
        print(f"time:{local_time[3]}:{local_time[4]}:{local_time[5]}")
        print(" experiment start")
        print("-" * 50)

        father_path = f"./capture/{local_time[0]}_{local_time[1]}_{local_time[2]}/"
        if not os.path.exists(father_path):
            os.mkdir(father_path)
        experiment_info_document = np.array([["plant_id: 1, chemical: N/A, position: N/A,"]])
        np.savetxt(father_path + "experiment_info.txt", experiment_info_document, fmt='%s')
        tip_id = 1
        np.savetxt(father_path + "tip_info.txt", np.array([tip_id]), fmt="%d")



    while plant_id < 121:
        print("plant id:", plant_id)
        chemical_id, exp_id, chemical_temp, tip_change_toggle = get_experiment_conf(requirement_doc, plant_id, chemical_temp)
        exp_target = experiment_pos[exp_id]
        print("target:", exp_target)
        print("chemical:", chemical_id)
        plant_info = [plant_id, chemical_id, exp_target]

        """
        change tip
        """
        if tip_change_toggle:
            tip_id = np.loadtxt(father_path + "tip_info.txt", dtype=int)
            if is_tip_attached:
                dispose_tip(tip_id)
                is_tip_attached = False

            succ_flag = False
            while not succ_flag:
                tip_id += 1
                total_err, succ_flag = attach_tip(tip_id, total_err)
                np.savetxt(father_path + "tip_info.txt", np.array([tip_id]), fmt="%d")
            is_tip_attached = True


        """
        detect plant
        """
        detect_pose = copy.deepcopy(plant_pose)
        detect_pose[0] -= 0.025
        robot_x.move_pose(temp_pose)
        robot_x.move_pose(detect_pose)
        time.sleep(.1)

        img_o, img_plant = rs_pipeline.get_plant_feature()
        center_pixel_list = None
        if exp_id == "C":
            plant_mask, center_pixel_list, img_mask = leaf_model.plant_center(img_plant)
        elif exp_id == "L":
            plant_mask, center_pixel_list, img_mask = leaf_model.leaf_biggest(img_plant)
        elif exp_id == "LL":
            plant_mask, center_pixel_list, img_mask = leaf_model.leaf_biggest_two(img_plant)
        if center_pixel_list is None:
            exp_target = experiment_pos[2]
            cv2.imwrite(father_path + f"plant_{plant_id}_origin.png", img_o)
            experiment_info = np.array([[f"plant_id: {plant_id}, chemical: {chemical_id}, position: NoDetection"]])
            experiment_info_document = np.append(experiment_info_document, experiment_info, axis=0)
            np.savetxt(father_path + "experiment_info.txt", experiment_info_document, fmt='%s')
            plant_id += 1
            continue

        """
        execute liquid dispensing 
        """
        for center_pixel in center_pixel_list:
            liquid_dispensing(chemical_id, plant_pose, center_pixel)

        plant_id += 1
        # input(f"tip_id:{tip_id}")
        detect_cts()

    dispose_tip(tip_id)
