import wrs.visualization.panda.world as wd
import wrs.robot_con.xarm_shuidi_grpc.xarm_shuidi_client as xsc
import cv2
import cv2.aruco as aruco
import numpy as np
from wrs import basis as rm, drivers as pk, robot_sim as xsm, modeling as gm
import time


# sensor handler
class SensorHandler:
    def __init__(self, pkx):
        self.pkx = pkx

    def get_marker_center(self):
        time.sleep(1)
        while True:
            self.pkx.device_get_capture()
            color_image_handle = self.pkx.capture_get_color_image()
            depth_image_handle = self.pkx.capture_get_depth_image()
            time.sleep(0.5)
            if color_image_handle and depth_image_handle:
                color_image = self.pkx.image_convert_to_numpy(color_image_handle)
                parameters = aruco.DetectorParameters_create()
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                corners, ids, rejectedImgPoints = aruco.detectMarkers(color_image,
                                                                      dictionary=aruco.Dictionary_get(
                                                                          aruco.DICT_4X4_250),
                                                                      parameters=parameters)
                aruco.drawDetectedMarkers(color_image, corners, ids=ids)
                cv2.imshow("test", color_image)
                cv2.waitKey(100)
                if ids is None:
                    return None
                image_xy = np.mean(np.mean(corners[0], axis=0), axis=0).astype(np.int16)
                pcd_pnt = self.pkx.transform_color_xy_to_pcd_xyz(input_color_image_handle=color_image_handle,
                                                            input_depth_image_handle=depth_image_handle,
                                                            color_xy=image_xy)
                self.pkx.image_release(depth_image_handle)
                self.pkx.capture_release()
                print(pcd_pnt)
                return pcd_pnt

    def get_point_cloud(self):
        while True:
            self.pkx.device_get_capture()
            color_image_handle = self.pkx.capture_get_color_image()
            depth_image_handle = self.pkx.capture_get_depth_image()
            if color_image_handle and depth_image_handle:
                point_cloud = self.pkx.transform_depth_image_to_point_cloud(depth_image_handle)
                self.pkx.image_release(depth_image_handle)
                self.pkx.capture_release()
                return point_cloud


pkx = pk.PyKinectAzure()
base = wd.World(cam_pos=[3, 3, 3], lookat_pos=[0, 0, 1])
gm.gen_frame().attach_to(base)
xss = xsm.XArmShuidi()
xsx = xsc.XArmShuidiClient(host="10.2.0.201:18300")
jnt_values = xsx.get_jnt_values()
xss.fk(component_name="arm", jnt_values=jnt_values)
xss.gen_meshmodel().attach_to(base)
# sensor handler
sensor_handler = SensorHandler(pkx)
# # calibrator
# dc = DepthCaliberator(robot_x=xsx, robot_s=xss)
# # start pos definition
# pos_start, rot_start = (np.array([-0.06960152, 0.01039783, 1.41780278]),
#                         np.array([[-1, 0, 0],
#                                   [0, -1, 0],
#                                   [0, 0, 1]]))
# joints = xss.ik(component_name="arm", tgt_pos=pos_start, tgt_rotmat=rot_start, max_n_iter=1000)
# xsx.move_jnts(component_name="arm", jnt_values=joints, time_intervals=1)
# pos = [np.array([0, 0, .2]),
#        np.array([0, 0, -.2]),
#        np.array([0, 0.2, 0]),
#        np.array([0, -0.2, 0]),
#        np.array([0.3, 0, .0]),
#        np.array([-0.1, 0, 0]),
#        np.array([0, 0.2, .2]),
#        np.array([0, -.2, .2]),
#        np.array([0.2, 0, .2]),
#        np.array([0.2, 0, -.2])]
# rotmat = [[np.array([0, 0, 1]), np.radians(10)]] * len(pos)
# action_pos_list = []
# action_rotmat_list = []
# for i in range(10):
#     action_pos_list.append(pos[i] + pos_start)
#     action_rotmat_list.append(np.dot(rm.rotmat_from_axangle(rotmat[i][0], rotmat[i][1]),
#                                      rot_start))
# # print(marker_pos_in_hnd)
# pos_arm_tcp, rot_arm_tcp = xss.arm.jlc.lnks[7]["gl_pos"], xss.arm.jlc.lnks[7]["gl_rotmat"]
# pos_hnd_tcp, rot_hand_tcp = xss.get_gl_tcp(manipulator_name="arm")
# # mgm.gen_frame(pos_arm_tcp, rot_arm_tcp).attach_to(base)
# # mgm.gen_frame(pos_hnd_tcp, rot_hand_tcp).attach_to(base)
# marker_pos_in_hnd = np.array([-0.026, 0, 0.069]) - np.array([0, 0, pos_hnd_tcp[2] - pos_arm_tcp[2]])
# # print("marker in hand", marker_pos_in_hnd)
# calibration_r = dc.calibrate(component_name="arm",
#                              sensor_marker_handler=sensor_handler,
#                              marker_pos_in_hnd=marker_pos_in_hnd,
#                              action_pos_list=action_pos_list,
#                              action_rotmat_list=action_rotmat_list)
# print(calibration_r)
# base.run()

# validate mph
from wrs.vision import load_calibration_data

affine_matrix, _, _ = load_calibration_data()
gm.GeometricModel(initor=rm.transform_points_by_homomat(
    affine_matrix,
    sensor_handler.get_point_cloud()
)).attach_to(base)

# # planner
# rrtc_planner = rrtc.RRTConnect(xss)
base.run()
