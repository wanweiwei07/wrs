import cv2.aruco as aruco
from wrs import drivers as hc, vision as rgb_uf

if __name__ == "__main__":
    # the path of the yaml file that saves the calibration parameters for camera rc0
    calib_file_path = './right_arm_camera_calib__.yaml'
    hcc = hc.Cam(host="192.168.0.60:18300")
    # get 10 images for detection
    # the detected results will be averaged using meanshift as the final pose
    img_list = []
    for i in range(10):
        img_list.append(hcc.getrc0img())
    # pose estimation
    pos, rot = rgb_uf.estimate_aruco_marker_pose(img_list=img_list,
                                                 calibration_data=calib_file_path,
                                                 aruco_marker_size=50,
                                                 aruco_dict=aruco.DICT_4X4_250)