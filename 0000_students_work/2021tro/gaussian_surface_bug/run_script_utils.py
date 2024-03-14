import copy
import os
import pickle
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from direct.stdpy import threading

import config
import utils.pcd_utils as pcdu
import utils.run_utils as ru
from localenv import envloader as el


class ForceRecorder(object):
    def __init__(self, motion_planner_x, f_id):
        self.motion_planner_x = motion_planner_x
        self.f_name = "./log/force_armjnts_log_" + f_id + ".pkl"
        self.__rbtx = self.motion_planner_x.rbtx
        self.__thread_recorder = None
        self.__thread_plot = None
        self.__flag = True
        self.__plotflag = False
        self.__ploton = True
        self.__armname = self.motion_planner_x.armname
        self.__force = []
        # record
        self.__info = []
        self.__force = []
        self.__armjnts = []

        # zero force
        self.__rbtx.zerotcpforce(armname="lft")
        self.__rbtx.zerotcpforce(armname="rgt")

    def start_record(self):
        self.__rbtx.zerotcpforce(armname=self.__armname)

        def recorder():
            while self.__flag:
                self.__force.append(self.__rbtx.getinhandtcpforce(armname=self.__armname))
                self.__armjnts = self.motion_planner_x.get_armjnts()
                self.__info.append([self.__force, self.__armjnts, time.time()])
                self.__plotflag = True
                time.sleep(.1)

        def plot():
            fig = plt.figure(1)
            plt.ion()
            plt.show()
            plt.ylim((-10, 10))
            while self.__ploton:
                if self.__plotflag:
                    plt.clf()
                    # print(len(self.__force))
                    force = [l[:3] for l in self.__force]
                    torque = [l[3:] for l in self.__force]
                    # linear_distance = [np.linalg.norm(self.fixgoal - np.array(dis[1])) for dis in self.__tcp]

                    x = [0.02 * i for i in range(len(force))]
                    plt.xlim((0, max(x)))
                    plt.subplot(211)
                    plt.plot(x, force, label=["x", "y", "z"])
                    plt.legend("xyz", loc='upper left')
                    plt.subplot(212)
                    plt.plot(x, torque, label=["Rx", "Ry", "Rz"])
                    plt.pause(0.005)
                time.sleep(.1)
            plt.savefig(f"{time.time()}.png")
            pickle.dump(self.__info, open(self.f_name, "wb"))
            plt.close(fig)

        self.__thread_recorder = threading.Thread(target=recorder, name="recorder")
        self.__thread_recorder.start()
        self.__thread_plot = threading.Thread(target=plot, name="plot")
        self.__thread_plot.start()
        print("start finish")

    def finish_record(self):
        if self.__thread_recorder is None:
            return
        self.__flag = False
        self.__thread_recorder.join()
        self.__thread_recorder = None

        if self.__thread_plot is None:
            return
        self.__ploton = False
        time.sleep(.05)
        self.__plotflag = False
        self.__thread_plot.join()
        self.__thread_plot = None


def load_motion_f(folder_name, f_name, root=config.MOTIONSCRIPT_REL_PATH):
    return pickle.load(open(os.path.join(root, folder_name, f_name), "rb"))


def load_motion_sgl(motion, folder_name, id_list, root=config.MOTIONSCRIPT_REL_PATH):
    motion_dict = pickle.load(open(root + folder_name + motion + ".pkl", "rb"))
    value = []
    for id in id_list:
        try:
            value = motion_dict[id]
            motion_dict = value
        except:
            continue
    objrelpos, objrelrot, path = value
    return objrelpos, objrelrot, path


def get_script_id_dict_multiplepen(folder_name, root=config.MOTIONSCRIPT_REL_PATH):
    id_list = []
    f_dict = load_motion_f(folder_name, "cam2place_pen.pkl", root=root)
    for k, v in f_dict.items():
        print(k, list(v.keys()))

    id = int(input("Select a grasp id from above:"))
    id_list.append(id)

    id = int(input("Select a objmat4 id from above:"))
    id_list.append(id)

    return id_list


def load_motion_seq(motion_seq, folder_name, id_list, armname=None, root=config.MOTIONSCRIPT_REL_PATH):
    path = []
    objrelpos = None
    objrelrot = None
    for motion in motion_seq:
        try:
            if armname is None:
                path_dict_path = root + folder_name + motion + ".pkl"
            else:
                path_dict_path = root + folder_name + "_".join([armname, motion]) + ".pkl"
            path_dict = pickle.load(open(path_dict_path, "rb"))
            for id in id_list:
                try:
                    path_dict = path_dict[id]
                except:
                    continue
            objrelpos, objrelrot, path_temp = path_dict
            # print(motion, path_temp)
            path.extend(path_temp)
        except:
            print(motion, "failed!")
            continue
    return objrelpos, objrelrot, path


def input_script_id(folder_name, armname=None, root=config.MOTIONSCRIPT_REL_PATH):
    def __have_dict(dict_list):
        bool = False
        for item in dict_list:
            if type(item) == dict:
                bool = True
                break
        return bool

    def __get_common_key(dict_list):
        key_list = []
        dict_cnt = 0
        for item in dict_list:
            try:
                key_list.extend(list(item.keys()))
                dict_cnt += 1
            except:
                continue
        return [key for key, cnt in Counter(key_list).items() if cnt == dict_cnt]

    def __dict_list_update(dict_list, id):
        new_dict_list = []
        for path_dict in dict_list:
            try:
                v = path_dict[id]
            except:
                v = path_dict
            new_dict_list.append(v)
        return new_dict_list

    id_list = []
    path_dict_list = []

    for f_name in os.listdir(os.path.join(root, folder_name)):
        if f_name[:4] == "draw" or f_name[:4] == "time":
            continue
        if armname is None:
            path_dict_list.append(load_motion_f(folder_name, f_name, root=root))
        else:
            if f_name[:3] == armname:
                path_dict_list.append(load_motion_f(folder_name, f_name, root=root))

    while __have_dict(dict_list=path_dict_list):
        print(__get_common_key(path_dict_list))
        id = int(input("Select a id from the list above:"))
        id_list.append(id)
        path_dict_list = __dict_list_update(path_dict_list, id)
    return id_list


def setting_sim(stl_f_name, pos=(600, 0, 780)):
    if stl_f_name[-3:] != 'stl':
        stl_f_name += '.stl'
    objcm = el.loadObj(stl_f_name, pos=pos, rot=(0, 0, 0))
    objcm.reparentTo(base.render)


def setting_hand(phxilocator, pcd_f_name="/dataset/mph/a_lft_0.pkl"):
    amat = phxilocator.amat
    hand_pcd = pickle.load(open(config.ROOT + pcd_f_name, "rb"))
    hand_pcd = pcdu.trans_pcd(pcdu.remove_pcd_zeros(hand_pcd), amat)
    hand = pcdu.reconstruct_surface(hand_pcd)

    pen_pos = (800, 400, 880)
    pen_pcd = el.loadObjpcd('pen.stl', pos=pen_pos, rot=(30, -90, 0), toggledebug=False)
    # base.pg.genpointcloudnp(pen_pcd).reparentTo(base.render)
    pen, pen_w, pen_h, pen_origin_pos, pen_rot_angle = pcdu.get_std_convexhull(pen_pcd, origin="tip", color=(1, 1, 0),
                                                                               toggledebug=False, toggleransac=False)
    hand.reparentTo(base.render)
    pen.reparentTo(base.render)
    base.pggen.plotSphere(base.render, pen_origin_pos, radius=5, rgba=(0, 1, 0, 1))

    return pen


def setting_real_simple(phoxi_f_path, amat):
    grayimg, depthnparray_float32, pcd = ru.load_phxiinfo(phoxi_f_name=phoxi_f_path, load=True)
    pcd = np.array([p for p in pcdu.trans_pcd(pcd, amat) if 790 < p[2] < 1100])
    pcdu.show_pcd(pcd)


def setting_real(phxilocator, phoxi_f_path, pen_stl_f_name, paintingobj_stl_f_name, resolution=1):
    pen_item = ru.get_obj_from_phoxiinfo_withmodel_nobgf(phxilocator, pen_stl_f_name, phoxi_f_name=phoxi_f_path,
                                                         x_range=(600, 1000), y_range=(200, 600), z_range=(810, 1000))
    if paintingobj_stl_f_name is not None:
        paintingobj_item = \
            ru.get_obj_from_phoxiinfo_withmodel_nobgf(phxilocator, paintingobj_stl_f_name, x_range=(400, 1080),
                                                      phoxi_f_name=phoxi_f_path, resolution=resolution)
    else:
        paintingobj_item = ru.get_obj_from_phoxiinfo_nobgf(phxilocator, phoxi_f_name=phoxi_f_path, load=True,
                                                           resolution=resolution)
    paintingobj_item.show_objcm(rgba=(1, 1, 1, .5))
    # paintingobj_item.show_objpcd(rgba=(1, 1, 0, 1))
    pen_item.show_objcm(rgba=(0, 1, 0, 1))


def show_drawmotion_ms(motion_planner, pen_cm, motion_f_path, grasp_id_list, jawwidth=20):
    draw_dict = pickle.load(open(motion_f_path, "rb"))
    objrelpos, objrelrot = None, None

    for grasp_id in grasp_id_list:
        path = []
        if grasp_id not in draw_dict.keys():
            continue
        for stroke_key, v in draw_dict[grasp_id].items():
            print("------", stroke_key, "--------")
            objrelpos, objrelrot, path_stroke = v
            if path != []:
                path_gotodraw = motion_planner.plan_start2end(start=path[-1], end=path_stroke[0])
            else:
                path_gotodraw = []
            path_up = motion_planner.get_moveup_path(path_stroke[-1], pen_cm, objrelpos, objrelrot, length=30)
            path.extend(path_gotodraw + path_stroke + path_up)
        motion_planner.ah.show_animation_hold(path, copy.deepcopy(pen_cm), objrelpos, objrelrot, jawwidth=jawwidth)


def show_drawmotion_ss(motion_planner, pen_cm, motion_f_path, grasp_id_list, jawwidth=20):
    draw_dict = pickle.load(open(motion_f_path, "rb"))

    for grasp_id in grasp_id_list:
        objrelpos, objrelrot, path = draw_dict[grasp_id]
        motion_planner.ah.show_animation_hold(path, copy.deepcopy(pen_cm), objrelpos, objrelrot, jawwidth=jawwidth)
