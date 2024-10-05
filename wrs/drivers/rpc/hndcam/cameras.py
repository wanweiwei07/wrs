import numpy as np
import os
import cv2
from panda3d.core import *
import cv2.aruco as aruco
import wrs.basis.robot_math as rm

class HndCam(object):

    def __init__(self, rgtcamid = [0,1], lftcamid = [2,3]):
        self.rgtcamid = rgtcamid
        self.lftcamid = lftcamid
        self.camcapid = self.rgtcamid[0]
        self.camcap = cv2.VideoCapture(self.camcapid+cv2.CAP_DSHOW)

    def getrc0img(self):
        if self.camcapid != self.rgtcamid[0]:
            self.camcap.release()
            self.camcapid = self.rgtcamid[0]
            self.camcap = cv2.VideoCapture(self.camcapid+cv2.CAP_DSHOW)
        self.camcap.read()
        return self.camcap.read()[1]

    def getrc1img(self):
        if self.camcapid != self.rgtcamid[1]:
            self.camcap.release()
            self.camcapid = self.rgtcamid[1]
            self.camcap = cv2.VideoCapture(self.camcapid+cv2.CAP_DSHOW)
        self.camcap.read()
        return self.camcap.read()[1]

    def getlc0img(self):
        if self.camcapid != self.lftcamid[0]:
            self.camcap.release()
            self.camcapid = self.lftcamid[0]
            self.camcap = cv2.VideoCapture(self.camcapid+cv2.CAP_DSHOW)
        self.camcap.read()
        return self.camcap.read()[1]

    def getlc1img(self):
        if self.camcapid != self.lftcamid[1]:
            self.camcap.release()
            self.camcapid = self.lftcamid[1]
            self.camcap = cv2.VideoCapture(self.camcapid+cv2.CAP_DSHOW)
        self.camcap.read()
        return self.camcap.read()[1]


if __name__ == "__main__":
    import time
    hdc = HndCam()
    while True:
        # ifnpa = pickle.loads(rkint.path.getifarray())
        # clnpa = pickle.loads(rkint.path.getclarray())
        # dnpa = pickle.loads(rkint.path.getrcimg())
        dnpa = hdc.getrc0img()
        cv2.imshow("Depth", dnpa)
        cv2.waitKey(100)
        # dnpa1 = hdc.getlc1img()
        # cv2.imshow("Depth", dnpa1)
        # cv2.waitKey(200)
        # cv2.imwrite('test.jpg',dnpa)