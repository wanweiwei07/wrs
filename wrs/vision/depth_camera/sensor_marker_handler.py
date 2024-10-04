from wrs import drivers as pcdt
import cv2.aruco as aruco
import numpy as np

class SensorMarkerHandler(object):

    def __init__(self):
        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.sensor_client = pcdt.PhxClient(host="192.168.125.100:18300")
        self.aruco_target_id_list = [0,1]

    def get_marker_center(self):
        self.sensor_client.triggerframe()
        img = self.sensor_client.gettextureimg()
        pcd = self.sensor_client.getpcd()
        width = img.shape[1]
        # detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_parameters)
        if len(corners) < len(self.aruco_target_id_list) or len(ids) != len(self.aruco_target_id_list):
            return None
        if ids[0] not in self.aruco_target_id_list or ids[1] not in self.aruco_target_id_list:
            return None
        center = np.mean(np.mean(corners, axis=0), axis=1)[0].astype(np.int32)
        marker_pos_in_sensor = pcd[width * center[1] + center[0]]
        return marker_pos_in_sensor