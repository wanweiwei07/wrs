import cv2
from wrs import drivers as pk, modeling as gm
import wrs.visualization.panda.world as wd
import os
import glob
import pickle
import numpy as np
import copy

imgid = 0
foldername = "./camimgs" + str(imgid) + "/"
if not os.path.exists(foldername):
    os.mkdir(foldername)
else:
    files = glob.glob(foldername + "*")
    for f in files:
        os.remove(f)
type = 'jpg'

base = wd.World(cam_pos=[0, 0, -10], lookat_pos=[0, 0, 10])
gm.gen_frame().attach_to(base)
pk_obj = pk.PyKinectAzure()
pcd_list = []
last_image = None
while True:
    pk_obj.device_get_capture()
    depth_image_handle = pk_obj.capture_get_depth_image()
    diff_image = None
    if depth_image_handle:
        depth_image = pk_obj.image_convert_to_numpy(depth_image_handle)
        # depth_image=cv2.erode(depth_image,np.ones((3,3)))
        # cv2.imwrite(foldername + 'opencv' + str(imgid) + '.' + end_type, depth_image)
        print("Saving an image...ID" + str(imgid))
        imgid += 1
        if last_image is not None:
            tmp_diff_image = np.abs(depth_image.astype(np.int32) - last_image.astype(np.int32))
            diff_image = copy.deepcopy(depth_image)
            diff_image[tmp_diff_image < 100] = 0
            diff_image[depth_image == 0] = 0
            # diff_image = cv2.erode(diff_image, np.ones((10,10)))
            diff_image_cmap = cv2.applyColorMap(np.round(diff_image / 30).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(foldername + '1opencv' + str(imgid) + '.' + type, diff_image_cmap)
            depth_image_cmap = cv2.applyColorMap(np.round(depth_image / 30).astype(np.uint8), cv2.COLORMAP_JET)
            last_image_cmap = cv2.applyColorMap(np.round(last_image / 30).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(foldername + '2opencv' + str(imgid) + '.' + type, depth_image_cmap)
            cv2.imshow('The images...', diff_image_cmap)
            cv2.imshow('The current image...', depth_image_cmap)
            cv2.imshow('The last image...', last_image_cmap)
        else:  # only same the first image as last image
            last_image = copy.deepcopy(depth_image)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
        # point_cloud = pk_obj.transform_depth_image_to_point_cloud(depth_image_handle)
        # pcd_list.append(point_cloud)
        # # pk_obj.image_release(color_image_handle)
        # # pk_obj.image_release(depth_image_handle)
        # # print("x")
        # cv2.imshow("", color_image)
    pk_obj.capture_release()
    # key = cv2.waitKey(1)
    # print(key)
    # if key == 32:
    #     break
pickle.dump(pcd_list, open("pcd_list.pkl", "wb"))
# base.run()
