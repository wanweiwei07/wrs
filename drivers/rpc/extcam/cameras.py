import numpy as np
import os
import cv2
from panda3d.core import *
import utiltools.robotmath as rm
import cv2.aruco as aruco

class ExtCam(object):

    def __init__(self):
        self.camcapid = 0
        self.camcap = cv2.VideoCapture(self.camcapid+cv2.CAP_DSHOW)
        self.camcap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.camcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # self.camcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.camcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camcap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.camcap.set(cv2.CAP_PROP_FOCUS, 0)

    def getimg(self):
        self.camcap.read()
        return self.camcap.read()[1]

if __name__ == "__main__":
    import time
    ec = ExtCam()
    while True:
        # ifnpa = pickle.loads(rkint.root.getifarray())
        # clnpa = pickle.loads(rkint.root.getclarray())
        # dnpa = pickle.loads(rkint.root.getrcimg())
        dnpa = ec.getimg()
        cv2.imshow("Depth", dnpa)
        cv2.waitKey(100)
        # dnpa1 = hdc.getlc1img()
        # cv2.imshow("Depth", dnpa1)
        # cv2.waitKey(200)
        # cv2.imwrite('test.jpg',dnpa)