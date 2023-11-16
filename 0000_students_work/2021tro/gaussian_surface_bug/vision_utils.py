import os
import pickle

import cv2
import numpy as np
import open3d as o3d
# import pyrealsense2 as rs2

import config


def load_kntcalibmat(amat_path=os.path.join(config.ROOT, "./camcalib/data/"), f_name="knt_calibmat.pkl"):
    amat = pickle.load(open(amat_path + f_name, "rb"))
    return amat


def map_depth2pcd(depthnarray, pcd):
    pcdnarray = np.array(pcd)
    depthnarray = depthnarray.flatten().reshape((depthnarray.shape[0] * depthnarray.shape[1], 1))

    pcd_result = []
    for i in range(len(depthnarray)):
        if depthnarray[i] != 0:
            pcd_result.append(pcdnarray[i])
        else:
            pcd_result.append(np.array([0, 0, 0]))

    pcdnarray = np.array(pcd_result)

    return pcdnarray


# def convert_depth2pcd(depthnarray):
#     h, w = depthnarray.shape
#     y_ = np.linspace(1, h, h)
#     x_ = np.linspace(1, w, w)
#     mesh_x, mesh_y = np.meshgrid(x_, y_)
#     z_ = depthnarray.flatten()
#     mph = np.zeros((np.size(mesh_x), 3))
#     mph[:, 0] = np.reshape(mesh_x, -1)
#     mph[:, 1] = np.reshape(mesh_y, -1)
#     mph[:, 2] = np.reshape(z_, -1)
#     return np.delete(mph, np.where(mph[:, 2] == 0)[0], axis=0)

def convert_depth2pcd(depthnarray, toggledebug=False):
    intr = pickle.load(open(os.path.join(config.ROOT, "gaussian_surface_bug", "realsense_intr.pkl"), "rb"))
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr["width"], intr["height"],
                                                                 intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
    depthimg = o3d.geometry.Image(depthnarray)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depthimg, pinhole_camera_intrinsic, np.eye(4))

    if toggledebug:
        o3d.visualization.draw_geometries([pcd])
        print(np.asarray(pcd.points))
    return np.asarray(pcd.points) * 1000


def map_gray2pcd(grayimg, pcd):
    pcdnarray = np.array(pcd)
    grayimg = grayimg.flatten().reshape((grayimg.shape[0] * grayimg.shape[1], 1))

    pcd_result = []
    for i in range(len(grayimg)):
        if grayimg[i] != 0:
            pcd_result.append(pcdnarray[i])
        # else:
        #     pcd_result.append(np.array([0, 0, 0]))
    pcdnarray = np.array(pcd_result)

    return pcdnarray


def mask2gray(mask, grayimg):
    grayimg[mask == 0] = 0
    return grayimg


def pcd2gray(pcd, grayimg):
    return NotImplemented


def map_depth2gray(depthnarray, greyimg):
    greyimg[depthnarray == 0] = 0
    return greyimg


def map_grayp2pcdp(grayp, grayimg, pcd):
    return np.array([pcd[int(grayp[1] * grayimg.shape[1] + grayp[0])]])


def map_pcdpinx2graypinx(pcdpinx, grayimg):
    a, b = divmod(pcdpinx, grayimg.shape[1])
    return b, a + 1


def gray23channel(grayimg):
    return np.stack((grayimg,) * 3, axis=-1)


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb2binary(img, threshold=128):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)[1]


def binary2pts(binary):
    shape = binary.shape
    p_list = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if binary[i, j]:
                p_list.append([i, j])
    return p_list
