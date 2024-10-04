import math

import numpy as np
import pcd_utils as pcdu
from wrs import modeling as gm

TOGGLEDEBUG = False


class Item(object):
    def __init__(self, *args, **kwargs):
        self.__objcm = None
        if "reconstruct" in list(kwargs.keys()):
            self.__reconstruct = kwargs["reconstruct"]
        else:
            self.__reconstruct = False
        if "objmat4" in list(kwargs.keys()):
            self.__objmat4 = kwargs["objmat4"]
        else:
            self.__objmat4 = np.eye(4)
        if "obj_cmodel" in list(kwargs.keys()):
            self.__objcm = kwargs["obj_cmodel"]
            self.__pcd_std = pcdu.get_objpcd(kwargs["obj_cmodel"], objmat4=self.__objmat4)
            self.__w, self.__h = pcdu.get_pcd_w_h(self.__pcd_std)
        if "mph" in list(kwargs.keys()):
            self.__pcd = kwargs["mph"]
            self.__nrmls = pcdu.get_nrmls(kwargs["mph"], camera_location=(.8, -.2, 1.8), toggledebug=TOGGLEDEBUG)
            if self.__reconstruct:
                self.__objcm = pcdu.reconstruct_surface(self.__pcd, radii=[.005])
        else:
            self.__pcd, self.__nrmls = pcdu.get_objpcd_withnrmls(self.__objcm, sample_num=kwargs["sample_num"],
                                                                 objmat4=self.__objmat4, toggledebug=TOGGLEDEBUG)
        # filtered_pcd = []
        # filtered_nrmls = []
        # dirc_v = np.asarray((0, 0, 1))
        # dirc_h = np.asarray((-1, -1, 0))
        # for i, n in enumerate(self.__nrmls):
        #     if n.dot(dirc_v) / (np.linalg.norm(n) * np.linalg.norm(dirc_v)) < 0.1 and \
        #             n.dot(dirc_h) / (np.linalg.norm(n) * np.linalg.norm(dirc_h)) < 0.1:
        #         filtered_pcd.append(self.__pcd[i])
        #         filtered_nrmls.append(self.__nrmls[i])
        # self.__pcd, self.__nrmls = np.asarray(filtered_pcd), np.asarray(filtered_nrmls)

        self.__pcdcenter = np.array((np.mean(self.__pcd[:, 0]),
                                     np.mean(self.__pcd[:, 1]),
                                     np.mean(self.__pcd[:, 2])))
        if "drawcenter" in list(kwargs.keys()):
            self.__drawcenter = kwargs["drawcenter"]
        else:
            self.__drawcenter = self.__pcdcenter
        # self.__nrmls = [-n for n in self.__nrmls]

    @property
    def objcm(self):
        return self.__objcm

    @property
    def pcd(self):
        return self.__pcd

    @property
    def nrmls(self):
        return self.__nrmls

    @property
    def pcd_std(self):
        return self.__pcd_std

    @property
    def objmat4(self):
        return self.__objmat4

    @property
    def pcdcenter(self):
        return self.__pcdcenter

    @property
    def drawcenter(self):
        return self.__drawcenter

    def set_drawcenter(self, posdiff):
        self.__drawcenter = self.__drawcenter + np.asarray(posdiff)

    def set_objmat4(self, objmat4):
        self.__objmat4 = objmat4
        self.objcm.sethomomat(objmat4)

    def reverse_nrmls(self):
        self.__nrmls = [-n for n in self.__nrmls]

    def gen_colps(self, radius=30, max_smp=120, show=False):
        nsample = int(math.ceil(self.objcm.trimesh.area / (radius ** 2 / 3.0)))
        if nsample > max_smp:
            nsample = max_smp
        samples = self.objcm.sample_surface_even(self.objcm.trimesh, nsample)
        samples = pcdu.trans_pcd(samples, self.objmat4)
        if show:
            for p in samples:
                gm.gen_sphere(pos=p, rgba=(1, 1, 0, .2), radius=radius)
        return samples

    def gen_colps_top(self, show=False):
        col_ps = []
        ps = self.pcd
        x_range = (min([x[0] for x in ps]), max([x[0] for x in ps]))
        y_range = (min([x[1] for x in ps]), max([x[1] for x in ps]))
        step = 10
        for i in range(int(x_range[0]), int(x_range[1]) + 1, step):
            for j in range(int(y_range[0]), int(y_range[1]) + 1, step):
                ps_temp = np.asarray([p for p in ps if i < p[0] < i + step and j < p[1] < j + step])
                if len(ps_temp) != 0:
                    p = ps_temp[ps_temp[:, 2] == max([x[2] for x in ps_temp])][0]
                    col_ps.append(p)
        col_ps = np.asarray(col_ps)
        if show:
            for p in col_ps:
                gm.gen_sphere(pos=p, rgba=(1, 0, 0, 1), radius=20)
        return col_ps

    def show_objcm(self, rgba=(1, 1, 1, 1), show_localframe=False):
        # import copy
        # objmat4 = copy.deepcopy(self.objmat4)
        # objmat4[:3, :3] = np.eye(3)
        # self.__objcm.sethomomat(objmat4)
        self.__objcm.set_homomat(self.objmat4)
        self.__objcm.set_rgba([rgba[0], rgba[1], rgba[2], rgba[3]])
        if show_localframe:
            self.__objcm.show_local_frame()
        self.__objcm.attach_to(base)

    def show_objpcd(self, rgba=(1, 1, 1, 1)):
        pcdu.show_pcd(self.__pcd, rgba=rgba)
