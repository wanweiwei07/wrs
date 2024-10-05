import cv2
import numpy as np
from .pykinectazure import PyKinectAzure
from . import _k4a

mtx = np.array([[610.16101074, 0, 638.35681152], [0, 610.19384766, 367.82455444], [0, 0, 1]])


def get_images(pk_obj):
    while True:
        pk_obj.device_get_capture()
        color_image_handle = pk_obj.capture_get_color_image()
        depth_image_handle = pk_obj.capture_get_depth_image()
        if color_image_handle and depth_image_handle:
            color_image = pk_obj.image_convert_to_numpy(color_image_handle)
            depth_image = pk_obj.transform_depth_to_color(depth_image_handle, color_image_handle)
            pk_obj.image_release(color_image_handle)
            pk_obj.image_release(depth_image_handle)
            pk_obj.capture_release()
            return color_image, depth_image
        else:
            print("get color image failed")
        pk_obj.capture_release()


def depth_to_point(mtx, pixel_pos, depth_image):
    img_x, img_y = pixel_pos
    rgbcam_fx = mtx[0][0]
    rgbcam_fy = mtx[1][1]
    rgbcam_cx = mtx[0][2]
    rgbcam_cy = mtx[1][2]
    world_x = (img_x - rgbcam_cx) * depth_image / (rgbcam_fx * 1000)
    world_y = (img_y - rgbcam_cy) * depth_image / (rgbcam_fy * 1000)
    world_z = depth_image / 1000
    return np.array([world_x, world_y, world_z])


def pcd_read(depth_image, mtx):
    image_size = depth_image.shape
    rgbcam_fx = mtx[0][0]
    rgbcam_fy = mtx[1][1]
    rgbcam_cx = mtx[0][2]
    rgbcam_cy = mtx[1][2]
    length = image_size[0] * image_size[1]
    kx = (np.arange(image_size[1]) - rgbcam_cx) / (rgbcam_fx * 1000)
    kx = np.tile(kx, image_size[0])
    ky = (np.arange(image_size[0]) - rgbcam_cy) / (rgbcam_fy * 1000)
    ky = ky.repeat(image_size[1])
    k = np.array(list(zip(kx, ky, np.ones(length, dtype=int) / 1000)))
    depth = depth_image.repeat(3).reshape(length, 3) + \
            np.tile(np.array([rgbcam_fx, rgbcam_fy, 0]), length).reshape(length, 3)
    return k * depth


if __name__ == "__main__":
    pk_obj = PyKinectAzure()
    pk_obj.device_open()
    device_config = pk_obj.config
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
    print(device_config)
    pk_obj.device_start_cameras(device_config)
    color, depth = get_images(pk_obj)
    cv2.imshow("color", color)
    cv2.imshow("depth", depth)
    cv2.waitKey(0)
