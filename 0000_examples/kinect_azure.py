import cv2
# import cv2.aruco as aruco
from wrs import wd, rm, mgm
import wrs.drivers.devices.kinect_azure.pykinectazure as pka

base = wd.World(cam_pos=rm.vec(0, 0, -10), lookat_pos=rm.vec(0, 0, 10))
mgm.gen_frame().attach_to(base)
# base.run()
pk_obj = pka.PyKinectAzure()
pk_obj.device_open()
pk_obj.device_start_cameras()
base.run()

pcd_list = []
marker_center_list = []
def update(pk_obj, pcd_list, marker_center_list, task):
    if len(pcd_list) != 0:
        for pcd in pcd_list:
            pcd.detach()
        pcd_list.clear()
    if len(marker_center_list) != 0:
        for marker_center in marker_center_list:
            marker_center.detach()
        marker_center_list.clear()
    pk_obj.device_get_capture()
    color_image_handle = pk_obj.capture_get_color_image()
    depth_image_handle = pk_obj.capture_get_depth_image()
    if color_image_handle and depth_image_handle:
        color_image = pk_obj.image_convert_to_numpy(color_image_handle)
        point_cloud = pk_obj.transform_depth_image_to_point_cloud(depth_image_handle)
        parameters = aruco.DetectorParameters_create()
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(color_image,
                                                              dictionary=aruco.Dictionary_get(aruco.DICT_4X4_250),
                                                              parameters=parameters)
        if len(corners) == 0:
            return task.cont
        image_xy = np.mean(np.mean(corners[0], axis=0), axis=0).astype(np.int16)
        pcd_pnt = pk_obj.transform_color_xy_to_pcd_xyz(input_color_image_handle=color_image_handle,
                                                       input_depth_image_handle=depth_image_handle,
                                                       color_xy=image_xy)
        # cv2.circle(color_image, tuple(image_xy), 10, (255, 0, 0), -1)
        # cv2.imshow("test", color_image)
        # cv2.waitKey(0)
        mypoint_cloud = gm.GeometricModel(initor=point_cloud)
        mypoint_cloud.attach_to(base)
        pcd_list.append(mypoint_cloud)
        marker_center = gm.gen_sphere(pos= pcd_pnt, radius=.1)
        marker_center.attach_to(base)
        marker_center_list.append(marker_center)
        pk_obj.image_release(color_image_handle)
        pk_obj.image_release(depth_image_handle)
    pk_obj.capture_release()
    return task.cont

taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[pk_obj, pcd_list, marker_center_list],
                      appendTask=True)
base.run()
