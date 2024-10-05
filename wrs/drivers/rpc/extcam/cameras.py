import numpy as np
import os
import cv2
from panda3d.core import *
import cv2.aruco as aruco
import wrs.basis.robot_math as rm

class ExtCam(object):

    def __init__(self):
        self.cam_id = 0
        self.cam_cap = cv2.VideoCapture(self.cam_id+cv2.CAP_DSHOW)
        self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # self.cam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cam_cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cam_cap.set(cv2.CAP_PROP_FOCUS, 0)

    def getimg(self):
        self.cam_cap.read()
        return self.cam_cap.read()[1]

if __name__ == "__main__":
    import time
    ec = ExtCam()
    while True:
        # ifnpa = pickle.loads(rkint.path.getifarray())
        # clnpa = pickle.loads(rkint.path.getclarray())
        # dnpa = pickle.loads(rkint.path.getrcimg())
        dnpa = ec.getimg()
        cv2.imshow("Depth", dnpa)
        cv2.waitKey(100)
        # dnpa1 = hdc.getlc1img()
        # cv2.imshow("Depth", dnpa1)
        # cv2.waitKey(200)
        # cv2.imwrite('test.jpg',dnpa)