import robothelper
import numpy as np
import cv2
import pickle
import environment.collisionmodel as cm

if __name__ == '__main__':
    yhx = robothelper.RobotHelperX(usereal=False)
    yhx.pxc.triggerframe()
    bgdepth = yhx.pxc.getdepthimg()
    pickle.dump(bgdepth, open("../../databackground/bgdepthlow.pkl", "wb"))
    bgpcd = yhx.pxc.getpcd()
    pickle.dump(bgpcd, open("../../databackground/bgpcddepthlow.pkl", "wb"))