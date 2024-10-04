import copy
import os
import pickle
from random import choice

import cv2
import numpy as np
import sklearn.cluster as skc

import config
# import trimesh.sample as ts
import pcd_utils as pcdu
import phoxi as phoxi
import vision_utils as vu
from wrs import basis as rm, modeling as cm
# import utiltools.thirdparty.o3dhelper as o3dh
import envloader as el


class PhxiLocator(object):
    def __init__(self, phxi, amat_f_name=config.AMAT_F_NAME):
        self.__phxi = phxi
        # temporal data structure for acceleration
        self.__processed = None
        self.amat = pickle.load(open(os.path.join(config.ROOT, "camcalib/data/" + amat_f_name), "rb"))

    def getallconnectedcomponents(self, depthnparray_float32, expandingdelta=5, minsize=300, toggledebug=False):
        """
        finds all connected components whose area is larger than minsize
        returns a list of nparray_float32

        :param depthnparray_float32:
        :param expandingdelta:
        :param minsize:
        :return:

        author: weiwei
        date: 20191207
        """

        depthnparray_float32_cp = copy.deepcopy(depthnparray_float32)
        components_depthfloat32list = []
        while True:
            tmpidx = np.nonzero(depthnparray_float32_cp)
            if len(tmpidx[0]) == 0:
                break
            seed = [tmpidx[0][0], tmpidx[1][0]]
            tmpcomponent = self.__region_growing(depthnparray_float32_cp, seed, expandingdelta)
            depthnparray_float32_cp[tmpcomponent != 0] = 0
            if np.count_nonzero(tmpcomponent) > minsize:
                components_depthfloat32list.append(tmpcomponent)
            else:
                continue
            if toggledebug:
                # cv2.imshow("getting connected component",
                #            self.__phoxi.scalefloat32uint8(components_depthfloat32list[-1]))
                cv2.imshow("getting connected component", components_depthfloat32list[-1])
                cv2.waitKey(0)
                # cv2.imshow("getting connected component", self.__phoxi.scalefloat32uint8(depthnparray_float32_cp))
                cv2.imshow("getting connected component", depthnparray_float32_cp)
                cv2.waitKey(0)
        return components_depthfloat32list

    def getnextconnectedcomponent(self, depthnparray_float32, expandingdelta=5, minsize=300):
        """
        finds the next connected components whose area is larger than minsize
        returns one nparray_float32

        :param depthnparray_float32:
        :param expandingdelta:
        :param minsize:
        :return:

        author: weiwei
        date: 20191207
        """

        depthnparray_float32_cp = copy.deepcopy(depthnparray_float32)
        while True:
            tmpidx = np.nonzero(depthnparray_float32_cp)
            if len(tmpidx[0]) == 0:
                return np.zeros_like(depthnparray_float32)
            seed = [tmpidx[0][0], tmpidx[1][0]]
            tmpcomponent = self.__region_growing(depthnparray_float32_cp, seed, expandingdelta)
            depthnparray_float32_cp[tmpcomponent != 0] = 0
            if np.count_nonzero(tmpcomponent) > minsize:
                return tmpcomponent
            else:
                continue

    def getnextconnectedcomponent_specifyseed(self, depthnparray_float32, seed=[0, 0], expandingdelta=5, minsize=300):
        """
        finds the next connected components whose area is larger than minsize
        region grow using the given seed
        returns one nparray_float32

        :param depthnparray_float32:
        :param expandingdelta:
        :param minsize:
        :return:
        """

        depthnparray_float32_cp = copy.deepcopy(depthnparray_float32)
        while True:
            tmpidx = np.nonzero(depthnparray_float32_cp)
            if len(tmpidx[0]) == 0:
                return np.zeros_like(depthnparray_float32)
            tmpcomponent = self.__region_growing(depthnparray_float32_cp, seed, expandingdelta)
            depthnparray_float32_cp[tmpcomponent != 0] = 0
            if np.count_nonzero(tmpcomponent) > minsize:
                return tmpcomponent
            else:
                continue

    def remove_pcd_bg(self, pcd, bg_path=config.ROOT + "/img/background/", bg_f_name="bg_0.pkl"):
        with open(bg_path + bg_f_name, "rb") as file:
            pcd_bg = pickle.load(file)[2]
        pcd_input = pcdu.trans_pcd(pcd, self.amat)

        pcd_fg = []
        for id, point in enumerate(pcd_input):
            x, y, z, _ = point

            if z > pcd_bg[id][2] + 3:
                pcd_fg.append([x, y, z])
        return np.array(pcd_fg)

    def remove_depth_bg(self, depthnparray_float32, bg_f_name="bg_0.pkl", bg_path=config.ROOT + "/img/background/",
                        x_range=None, y_range=None, toggledebug=False):
        with open(bg_path + bg_f_name, "rb") as f:
            depthnarray_bg = pickle.load(f)[1]

        depthnarray_fg = np.zeros_like(depthnparray_float32)
        if x_range is None or y_range is None:
            depthnarray_fg[:, :] = -(depthnparray_float32[:, :] - (depthnarray_bg[:, :] + 2))
        else:
            depthnarray_range = np.zeros_like(depthnarray_bg)
            depthnarray_range[x_range[0]:x_range[1], y_range[0]:y_range[1]] = depthnarray_bg[x_range[0]:x_range[1],
                                                                              y_range[0]:y_range[1]]
            depthnarray_fg[x_range[0]:x_range[1], y_range[0]:y_range[1]] = -(
                    depthnparray_float32[x_range[0]:x_range[1], y_range[0]:y_range[1]] - (
                    depthnarray_range[x_range[0]:x_range[1], y_range[0]:y_range[1]] + 2))

        depthnarray_fg[depthnarray_fg < 5] = 0
        depthnarray_fg[depthnarray_fg > 50] = 0

        maxdepth = 50
        mindepth = 1
        fakezero = 1
        depthnarray_fg = np.zeros_like(depthnarray_fg).astype(dtype=np.uint8)
        depthnarray_fg[depthnarray_fg != 0] = (depthnarray_fg[depthnarray_fg != 0] - mindepth) / (
                maxdepth - mindepth) * (255 - fakezero) + fakezero
        if toggledebug:
            cv2.imshow("tst", depthnarray_fg)
            cv2.waitKey(0)

        return depthnarray_fg

    def find_objpcd_list_by_pos(self, pcd, x_range=(200, 800), y_range=(0, 600), z_range=(790, 1000), eps=5,
                                toggledebug=False, scan_num=1):
        real_pcd = pcdu.trans_pcd(pcd, self.amat)
        # pcdu.show_pcd([p for p in real_pcd if p[2] < 900], rgba=(.5, .5, .5, .1))
        # base.run()
        pcd_result = []
        for p in real_pcd:
            if x_range[0] < p[0] < x_range[1] and y_range[0] < p[1] < y_range[1] and z_range[0] < p[2] < z_range[1]:
                pcd_result.append(p)
        pcd_result = np.array(pcd_result)
        db = skc.DBSCAN(eps=eps, min_samples=50 * scan_num).fit(pcd_result)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("n_clusters:", n_clusters)
        unique_labels = set(labels)
        objpcd_list = []
        for k in unique_labels:
            if k == -1:
                continue
            else:
                class_member_mask = (labels == k)
                temppartialpcd = pcd_result[class_member_mask & core_samples_mask]
                if len(temppartialpcd) > 500:
                    objpcd_list.append(temppartialpcd)
        if toggledebug:
            # pcdu.show_pcd(real_pcd, rgba=(1, 1, 1, .1))
            pcdu.show_pcd(pcd_result, rgba=(1, 1, 0, 1))
            for objpcd in objpcd_list:
                pcdu.show_pcd_withrbt(objpcd, rgba=(choice([0, 1]), choice([0, 1]), 1, 1))
            base.run()
        return objpcd_list

    def find_closest_objpcd_by_stl(self, src_stl_f_name, objpcd_list, inithomomat=None, use_rmse=True):
        objcm = cm.CollisionModel(objinit=os.path.join(config.ROOT + '/obstacles/' + src_stl_f_name))
        min_rmse = 100
        max_fitness = 0
        result_pcd = None
        print("---------------find closest objpcd---------------")
        print("find obj:", src_stl_f_name.split(".stl")[0])

        for tgt in objpcd_list:
            tgt_narray = np.array(tgt)
            if inithomomat is None:
                inithomomat = self.__match_pos(np.asarray(ts.sample_surface(objcm.trimesh, count=10000)), tgt_narray)
            src_narray = pcdu.get_objpcd_partial_bycampos(objcm, objmat4=inithomomat, sample_num=len(tgt_narray),
                                                          toggledebug=False)
            rmse, fitness = self.__get_icp_scores(tgt_narray, src_narray, show_icp=False)
            if use_rmse:
                print("rmse:", rmse, "mph axis_length", len(tgt))
                if 0.0 < rmse < min_rmse:
                    result_pcd = copy.deepcopy(tgt_narray)
                    min_rmse = rmse
            else:
                print("fitness:", fitness, "mph axis_length", len(tgt))
                if fitness != 0.0 and fitness > max_fitness:
                    result_pcd = copy.deepcopy(tgt_narray)
                    max_fitness = fitness

        return result_pcd

    def find_largest_objpcd(self, objpcd_list):
        max_length = 0
        result_pcd = None
        for objpcd in objpcd_list:
            if len(objpcd) > max_length:
                max_length = len(objpcd)
                result_pcd = objpcd
        print("largest mph axis_length:", max_length)
        return result_pcd

    def find_objdepth_list_by_size(self, sourcenparray_float32, expandingdelta=5, toggledebug=False,
                                   radius_range=(50, 100), wh_ratio=(0, 3)):
        objnparray_float32list = self.getallconnectedcomponents(sourcenparray_float32, expandingdelta,
                                                                toggledebug=False)
        print("num of obj detected:", len(objnparray_float32list))
        result = []

        for component in objnparray_float32list:
            cnts = cv2.findContours(component.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts[0]:
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)
                w, h = self.__get_minrec_w_h(c)
                if wh_ratio[0] < w / h < wh_ratio[1] and radius_range[0] < radius < radius_range[1]:
                    print("---------------------")
                    print("center:", center)
                    print("major_radius:", radius)
                    print("w:", w)
                    print("h:", h)
                    result.append(component)
                    sourcenparray_float32[component != 0] = 0
                    if toggledebug:
                        print("---------candidate--------")
                        print("center:", center)
                        print("major_radius:", radius)
                        print("w:", w)
                        print("h:", h)
                        cv2.imshow('Result', component)
                        cv2.waitKey()

        return result

    def find_hand_icp(self, sourcenparray_float32, pcd, expandingdelta=8, toggledebug=False, show_icp=False):
        objnparray_float32list = self.getallconnectedcomponents(sourcenparray_float32, expandingdelta,
                                                                toggledebug=False)
        print("num of obj detected:", len(objnparray_float32list))
        result = None
        min_rmse = 100

        for component in objnparray_float32list:
            cnts = cv2.findContours(component.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts[0]:
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)
                w, h = self.__get_minrec_w_h(c)
                if w / h < 3 and radius > 50:
                    print("---------------------")
                    print("center:", center)
                    print("major_radius:", radius)
                    print("w:", w)
                    print("h:", h)
                    rmse = self.__get_hand_icp_rmse(component, pcd, show_icp=show_icp)
                    print("rmse:", rmse)
                    if 0.0 < rmse < min_rmse:
                        min_rmse = rmse
                        print("result updated!")
                        result = component
                        sourcenparray_float32[component != 0] = 0
                    if toggledebug:
                        print("---------candidate--------")
                        print("center:", center)
                        print("major_radius:", radius)
                        print("w:", w)
                        print("h:", h)
                        print("rmse:", rmse)
                        cv2.imshow('Result', component)
                        cv2.waitKey()

        return result

    def find_objinhand_rg(self, tcppos, grayimg, depthnparray_float32, pcd, toggledebug=False):
        """
        region growing, return obj depthnarray

        :param tcppos:
        :param grayimg:
        :param depthnparray_float32:
        :param pcd:
        :param toggledebug:
        :return:
        """
        pcd = pcdu.trans_pcd(pcd, self.amat)
        base.pggen.plotSphere(base.render, tcppos, radius=5, rgba=(1, 0, 0, 1))

        tcp_pcd_idx, tcp_pcd = pcdu.get_pcdidx_by_pos(pcd, tcppos)
        seed = vu.map_pcdpinx2graypinx(tcp_pcd_idx, grayimg)

        grayimg_mask = self.__mask_by_seed(grayimg, center=seed, width=200)
        depthnparray_float32_mask = self.__mask_by_seed(depthnparray_float32, center=seed, width=100)

        # result = self.getnextconnectedcomponent_specifyseed(depthnparray_float32_mask, seed=seed,
        #                                                     expandingdelta=4)
        tmpcomponent_list = self.getallconnectedcomponents(depthnparray_float32_mask, expandingdelta=8,
                                                           toggledebug=False)
        result = self.get_highest_component_depthnparray(tmpcomponent_list, pcd)

        if toggledebug:
            cv2.circle(grayimg, seed, 2, (255, 0, 0), 4)
            cv2.imshow("gray", grayimg)
            cv2.waitKey(0)

            cv2.circle(grayimg_mask, seed, 2, (255, 0, 0), 4)
            cv2.imshow("gray", grayimg_mask)
            cv2.waitKey(0)

            cv2.imshow("depth", depthnparray_float32_mask)
            cv2.waitKey(0)
            cv2.imshow("result", result)
            cv2.waitKey(0)

            # pcdu.show_pcd(mph)
            base.pggen.plotSphere(base.render, tcppos, radius=10, rgba=(1, 0, 0, 1))
            base.pggen.plotSphere(base.render, tcp_pcd, radius=10, rgba=(0, 1, 0, 1))
            base.run()

        return result

    def find_objinhand_pcd(self, tcppos, pcd, src_stl_f_name, mode="icp", toggledebug=False):
        objpcd_list = self.find_objpcd_list_by_pos(pcd, x_range=(tcppos[0] - 100, tcppos[0] + 100),
                                                   y_range=(tcppos[1] - 120, tcppos[1] + 120),
                                                   z_range=(tcppos[2] - 50, tcppos[2] + 50), eps=5,
                                                   toggledebug=toggledebug)
        if mode == "icp":
            objpcd = self.find_closest_objpcd_by_stl(src_stl_f_name, objpcd_list)
        else:
            objpcd = self.find_largest_objpcd(objpcd_list)

        return objpcd

    def get_highest_component_depthnparray(self, depthnparray_list, pcd):
        highest_depth = 0
        highest_component = None
        for component in depthnparray_list:
            component_pcd = pcdu.remove_pcd_zeros(vu.map_depth2pcd(component, pcd))
            # pcdu.show_pcd(component_pcd,
            #               colors=(random.choice([0, 1]), random.choice([0, 1]), random.choice([0, 1]), 1))
            tmpdepth = np.mean(component_pcd[:, 2])
            # print("temp depth:", tmpdepth)
            if tmpdepth > highest_depth:
                highest_component = component
                highest_depth = tmpdepth
        return highest_component

    def match_pcdncm_ptpt(self, target, src_stl_f_name, dv=10, show_icp=False):
        obj = cm.CollisionModel(objinit=os.path.join(config.ROOT + '/obstacles/' + src_stl_f_name))
        source = np.asarray(ts.sample_surface(obj.trimesh, count=10000))
        source = source[source[:, 2] > 5]
        target = np.asarray(target)

        rmse, fitness, homomat = o3dh.registration_ptpt(source, target, downsampling_voxelsize=dv,
                                                        toggledebug=show_icp)
        return homomat

    def match_pcdncm(self, target, source_cm, inithomomat=None, match_rotz=False, toggledebug=False):
        target_pcd_center = pcdu.get_pcd_center(target)
        if inithomomat is None:
            inithomomat = self.__match_pos(np.asarray(ts.sample_surface(source_cm.trimesh, count=10000)), target)
        print("Length of target mph", len(target))
        source = pcdu.get_objpcd_partial_bycampos(source_cm, inithomomat, sample_num=len(target), toggledebug=False)
        # source = pcdu.get_objpcd(source_cm, objmat4=inithomomat, sample_num=len(target_pcd))

        rmse, fitness, homomat = o3dh.registration_icp_ptpt(source, target, inithomomat=np.eye(4), maxcorrdist=5,
                                                            toggledebug=toggledebug)
        # rmse, pos = o3du.registration_ptpt(source, target, downsampling_voxelsize=10, toggledebug=toggledebug)
        print("match rmse, fitness:", rmse, fitness)

        if match_rotz:
            min_rmse = rmse
            for rot in range(45, 180 + 1, 45):
                inithomomat_rotted = copy.deepcopy(inithomomat)
                inithomomat_rotted[:3, :3] = np.dot(inithomomat_rotted[:3, :3], rm.rodrigues([0, 0, 1], rot))

                temp_center = pcdu.get_pcd_center(pcdu.trans_pcd(source, inithomomat_rotted))
                inithomomat_rotted[:3, 3] = inithomomat_rotted[:3, 3] + (target_pcd_center - temp_center)

                rmse_rotted, fitness_rotted, homomat_rotted = \
                    o3dh.registration_icp_ptpt(source, target, inithomomat_rotted, maxcorrdist=5,
                                               toggledebug=toggledebug)
                print("rotmat rmse, fitness:", rot, rmse_rotted, fitness_rotted)

                if rmse_rotted < min_rmse and rmse_rotted != 0.0:
                    homomat = homomat_rotted
                    min_rmse = rmse_rotted

        result = np.dot(homomat, inithomomat)
        # if (inithomomat[:3, :3] == np.eye(3)).all():
        #     result = copy.deepcopy(result)
        #     result[:3, 0] = np.array([1, 0, 0])
        #     result[:3, 1] = np.array([0, 1, 0])
        print("---------------match mph&mcm done---------------")

        if toggledebug:
            show_cm = copy.deepcopy(source_cm)
            show_cm.sethomomat(inithomomat)
            show_cm.setColor(0, 1, 0, 0.5)
            show_cm.reparentTo(base.render)

            show_cm2 = copy.deepcopy(source_cm)
            show_cm2.sethomomat(homomat)
            show_cm2.setColor(0, 0, 1, 0.5)
            show_cm2.reparentTo(base.render)

            pcdu.show_pcd(source)
            pcdu.show_pcd(target)
            base.run()

        return result

    def __get_minrec_w_h(self, c):
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        return self.__sort_w_h(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))

    def __mask_by_seed(self, narray, center, width=50):
        narray_masked = np.zeros(narray.shape)
        narray_masked[center[1] - width:center[1] + width, center[0] - width:center[0] + width] = \
            narray[center[1] - width:center[1] + width, center[0] - width:center[0] + width]
        narray_masked = narray_masked.astype(type(narray[0][0][0]))
        return narray_masked

    # def __match_2d(self, target_pcd, table_h):
    #     target_pcd = o3du.nparray2o3dpcd(target_pcd)
    #     target_pcd_removed = o3du.removeoutlier(target_pcd, nb_points=50, major_radius=10)
    #     target_pcd = o3du.o3dpcd2nparray(target_pcd_removed)
    #
    #     target_2d = target_pcd[:, :2]  # TODO clip using sensor z
    #     ca = np.cov(target_2d, y=None, rowvar=False, bias=True)
    #     v, vect = np.linalg.eig(ca)
    #     tvect = np.transpose(vect)
    #
    #     # use the inverse of the eigenvectors as a rotation matrix and
    #     # rotate the points so they align with the x and y axes
    #     ar = np.dot(target_2d, np.linalg.inv(tvect))
    #     # get the minimum and maximum x and y
    #     mina = np.min(ar, axis=0)
    #     maxa = np.max(ar, axis=0)
    #     diff = (maxa - mina) * 0.5
    #     # the center is just half way between the min and max xy
    #     center = mina + diff
    #     # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    #     corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]],
    #                         center + [diff[0], diff[1]], center + [-diff[0], diff[1]],
    #                         center + [-diff[0], -diff[1]]])
    #     # use the the eigenvectors as a rotation matrix and
    #     # rotate the corners and the centerback
    #     corners = np.dot(corners, tvect)
    #     center = np.dot(center, tvect)
    #
    #     axind = np.argsort(v)
    #     pos = np.eye(4)
    #     pos[:3, axind[0]] = np.array([vect[0, 0], vect[1, 0], 0])
    #     pos[:3, axind[1]] = np.array([vect[0, 1], vect[1, 1], 0])
    #     pos[:3, 2] = np.array([0, 0, 1])
    #     if np.cross(pos[:3, 0], pos[:3, 1])[2] < -.5:
    #         pos[:3, 1] = -pos[:3, 1]
    #     pos[:3, 3] = np.array([center[0], center[1], table_h])
    #     return pos

    def __match_pos(self, source_pcd, target_pcd):
        source_center = pcdu.get_pcd_center(source_pcd)
        target_center = pcdu.get_pcd_center(target_pcd)
        diff = target_center - source_center
        rot = np.eye(3)
        return rm.homobuild(diff, rot)

    def __get_icp_scores(self, target, source, show_icp=False):
        inithomomat = self.__match_pos(source, target)

        rmse, fitness, _ = o3dh.registration_icp_ptpt(source, target, inithomomat, maxcorrdist=5, toggledebug=show_icp)
        # rmse, transform = o3d_helper.registration_ptpt(source, target, downsampling_voxelsize=10,
        #                                                toggledebug=show_icp)
        return rmse, fitness

    def __get_hand_icp_rmse(self, depthnparray_float32, pcd, show_icp=False):
        dv = 10
        obj_pcd = vu.map_depth2pcd(depthnparray_float32, pcd)
        target = pcdu.trans_pcd(obj_pcd, self.amat)
        min_rmse = 100
        sample_name_list = ["a_lft_0", "a_rgt_0"]
        for sample_name in sample_name_list:
            source = pickle.load(open(config.ROOT + "/dataset/sample_handpcd/" + sample_name + "_pcd.pkl", "rb"))
            rmse, fitness, transform = o3dh.registration_ptpt(source, target, downsampling_voxelsize=dv,
                                                              toggledebug=show_icp)
            if rmse < min_rmse:
                min_rmse = rmse
        return min_rmse

    def __region_growing(self, depthimg, seed, expandingdelta):
        # cv2.imshow("", depthimg)
        # cv2.waitKey(0)
        list = []
        outimg = np.zeros_like(depthimg).astype(dtype=np.uint8)
        list.append((seed[0], seed[1]))
        self.__processed = np.zeros_like(depthimg).astype(dtype=np.uint8)
        while len(list) > 0:
            pix = list[0]
            outimg[pix[0], pix[1]] = 255
            for coord in self.__get8n(pix[0], pix[1], depthimg.shape, self.__processed):
                newvalue = int(depthimg[coord[0], coord[1]][0])
                cmpvalue = int(depthimg[pix[0], pix[1]][0])
                if depthimg[coord[0], coord[1]] != 0 and abs(newvalue - cmpvalue) < expandingdelta:
                    outimg[coord[0], coord[1]] = 255
                    if self.__processed[coord[0], coord[1]] == 0:
                        list.append(coord)
                    self.__processed[coord[0], coord[1]] = 1
                    # if not coord in processed:
                    #     list.append(coord)
                    # processed.append(coord)
            list.pop(0)
            # cv2.imshow("progress", outimg)
            # cv2.waitKey(1)
        return outimg

    def __get8n(self, x, y, shape, processed):
        out = []
        maxx = shape[1] - 1
        maxy = shape[0] - 1
        # top left
        outx = min(max(x - 1, 0), maxx)
        outy = min(max(y - 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # top center
        outx = x
        outy = min(max(y - 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # top right
        outx = min(max(x + 1, 0), maxx)
        outy = min(max(y - 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # left
        outx = min(max(x - 1, 0), maxx)
        outy = y
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # right
        outx = min(max(x + 1, 0), maxx)
        outy = y
        out.append((outx, outy))
        # bottom left
        outx = min(max(x - 1, 0), maxx)
        outy = min(max(y + 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # bottom center
        outx = x
        outy = min(max(y + 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))
        # bottom right
        outx = min(max(x + 1, 0), maxx)
        outy = min(max(y + 1, 0), maxy)
        if processed[outx, outy] == 0:
            out.append((outx, outy))

        return out

    def __get_avg_component_depth(self, depthnparray_float32, component_array):
        tmparray = depthnparray_float32[component_array != 0]
        tmparray = tmparray[tmparray.nonzero()]
        return tmparray.mean()

    def __sort_w_h(self, a, b):
        if a > b:
            return a, b
        else:
            return b, a


if __name__ == '__main__':
    import config

    base, env = el.loadEnv_wrs()
    phxi = phoxi.Phoxi(host=config.PHOXI_HOST)
    phxilocator = PhxiLocator(phxi, amat_f_name=config.AMAT_F_NAME)
    # phoxi_f_path = "cylinder/cylinder_0217.pkl"
    # stl_f_name = "cylinder.stl"
    phoxi_f_path = "egg/egg_0303_0.pkl"
    stl_f_name = "egg.stl"

    bg_f_name = "/bg_" + phoxi_f_path.split("/")[1].split("_")[1][:4] + ".pkl"
    grayimg, depthnparray_float32, pcd = phxi.loadalldata(f_name="img/" + phoxi_f_path)
