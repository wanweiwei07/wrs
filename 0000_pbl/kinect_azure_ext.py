import cv2
import wrs.visualization.panda.world as wd
import cv2.aruco as aruco
import numpy as np
from wrs import basis as rm, drivers as pk, modeling as gm

base = wd.World(cam_pos=[0, 0, -10], lookat_pos=[0, 0, 10])
gm.gen_frame().attach_to(base)
# base.run()
pk_obj = pk.PyKinectAzure()
pk_obj.device_open()
pk_obj.device_start_cameras()

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
        parameters = aruco.DetectorParameters_create()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(color_image,
                                                              dictionary=aruco.Dictionary_get(aruco.DICT_4X4_250),
                                                              parameters=parameters)
        if len(corners) == 0:
            pk_obj.image_release(color_image_handle)
            pk_obj.image_release(depth_image_handle)
            pk_obj.capture_release()
            continue
        image_xy_list = []
        for corner_list in corners:
            image_xy_list.append(np.mean(np.mean(corner_list, axis=0), axis=0).astype(np.int16))
        flag = False
        pcd_pnt_list = []
        for image_xy in image_xy_list:
            pcd_pnt = pk_obj.transform_color_xy_to_pcd_xyz(input_color_image_handle=color_image_handle,
                                                           input_depth_image_handle=depth_image_handle,
                                                           color_xy=image_xy)
            if np.sum(np.abs(pcd_pnt)) < 1:
                pk_obj.image_release(color_image_handle)
                pk_obj.image_release(depth_image_handle)
                pk_obj.capture_release()
                flag = True
                break
            pcd_pnt_list.append(pcd_pnt)
        if flag or len(pcd_pnt_list) < 3:
            continue
        print(pcd_pnt_list)
        id_origin = 2
        id_x = 0
        id_minus_y = 3
        for i, id in enumerate(ids):
            if id == 2:
                origin_xyz = pcd_pnt_list[i]
            if id == 0:
                plusx_xyz = pcd_pnt_list[i]
            if id == 3:
                minusy_xyz = pcd_pnt_list[i]
        axis_x = rm.unit_vector(plusx_xyz - origin_xyz)
        axis_y = rm.unit_vector(origin_xyz - minusy_xyz)
        axis_z = rm.unit_vector(np.cross(axis_x, axis_y))
        origin = origin_xyz
        origin_rotmat = np.eye(3)
        origin_rotmat[:, 0] = axis_x
        origin_rotmat[:, 1] = axis_y
        origin_rotmat[:, 2] = axis_z
        pickle.dump([origin, origin_rotmat], open("origin.pkl", "wb"))
        gm.gen_frame(pos=origin_xyz, rotmat=origin_rotmat, axis_length=1, axis_radius=.03).attach_to(base)
        # for image_xy in image_xy_list:
        #     cv2.circle(color_image, tuple(image_xy), 10, (255, 0, 0), -1)
        #     cv2.imshow("test", color_image)
        # cv2.waitKey(0)
        mypoint_cloud = gm.GeometricModel(initor=point_cloud)
        mypoint_cloud.attach_to(base)
        for pcd_pnt in pcd_pnt_list:
            marker_center = gm.gen_sphere(pos=pcd_pnt, radius=.1)
            marker_center.attach_to(base)
        base.run()
        pk_obj.image_release(color_image_handle)
        pk_obj.image_release(depth_image_handle)
    pk_obj.capture_release()

base.run()
