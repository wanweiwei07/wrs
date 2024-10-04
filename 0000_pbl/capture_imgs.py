import cv2
from wrs import drivers as pk, modeling as gm
import wrs.visualization.panda.world as wd
import os
import glob
import pickle

imgid = 0
foldername = "./camimgs" + str(imgid) + "/"
if not os.path.exists(foldername):
    os.mkdir(foldername)
else:
    files = glob.glob(foldername + "*")
    for f in files:
        os.remove(f)
type='jpg'

base = wd.World(cam_pos=[0, 0, -10], lookat_pos=[0, 0, 10])
gm.gen_frame().attach_to(base)
pk_obj = pk.PyKinectAzure()
pcd_list = []
while True:
    pk_obj.device_get_capture()
    color_image_handle = pk_obj.capture_get_color_image()
    if color_image_handle:
        color_image = pk_obj.image_convert_to_numpy(color_image_handle)
        cv2.imwrite(foldername + 'opencv' + str(imgid) + '.' + type, color_image)
        print("Saving an image...ID" + str(imgid))
        imgid += 1
        cv2.imshow('The images...', color_image)
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