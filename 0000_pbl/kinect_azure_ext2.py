import wrs.visualization.panda.world as wd
import pickle
import numpy as np
from wrs import basis as rm, drivers as pk, modeling as gm

origin, origin_rotmat = pickle.load(open("origin.pkl", "rb"))
origin_homomat = np.eye(4)
origin_homomat[:3, :3] = origin_rotmat
origin_homomat[:3, 3] = origin
print(origin, origin_rotmat)
# base = wd.World(cam_pos=origin, lookat_pos=origin_rotmat[:,0]-origin_rotmat[:,1]+origin_rotmat[:,2])
base = wd.World(cam_pos=np.zeros(3), lookat_pos=np.array([0, 0, 10]))
gm.gen_frame(axis_length=1, axis_radius=.1).attach_to(base)
# base.run()
pk_obj = pk.PyKinectAzure()
pk_obj.device_open()
pk_obj.device_start_cameras()
gm.gen_frame(pos=origin, rotmat=origin_rotmat, axis_length=1, axis_radius=.03).attach_to(base)
# base.run()
# pcd_list = []
# marker_center_list = []
# def update(pk_obj, pcd_list, marker_center_list, task):
#     if len(pcd_list) != 0:
#         for mph in pcd_list:
#             mph.detach()
#         pcd_list.clear()
#     if len(marker_center_list) != 0:
#         for marker_center in marker_center_list:
#             marker_center.detach()
#         marker_center_list.clear()
while True:
    pk_obj.device_get_capture()
    color_image_handle = pk_obj.capture_get_color_image()
    depth_image_handle = pk_obj.capture_get_depth_image()
    if color_image_handle and depth_image_handle:
        color_image = pk_obj.image_convert_to_numpy(color_image_handle)
        # cv2.imshow("test", color_image)
        # cv2.waitKey(0)
        point_cloud = pk_obj.transform_depth_image_to_point_cloud(depth_image_handle)
        point_cloud = rm.transform_points_by_homomat(rm.homomat_inverse(origin_homomat), point_cloud)
        point_cloud[point_cloud[:,0]<-1]=point_cloud[point_cloud[:,0]<-1]*0
        mypoint_cloud = gm.GeometricModel(initor=point_cloud)
        mypoint_cloud.attach_to(base)
        base.run()
        pk_obj.image_release(color_image_handle)
        pk_obj.image_release(depth_image_handle)
    pk_obj.capture_release()

base.run()
