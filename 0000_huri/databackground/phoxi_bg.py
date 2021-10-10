import robothelper
import numpy as np
import cv2
import pickle
import utiltools.robotmath as rm
import environment.collisionmodel as cm

if __name__ == '__main__':
    rhx = robothelper.RobotHelperX(usereal=True)

    eepos = np.array([400, 0, 200])
    eerot=np.array([[1,0,0],[0,0,-1],[0,1,0]]).T

    armjnts= rhx.movetoposrot(eepos = eepos, eerot=eerot, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    rhx.pxc.triggerframe()
    bgdepth = rhx.pxc.getdepthimg()
    pickle.dump(bgdepth, open("../databackground/tro_bgdepthlow1.pkl", "wb"))
    bgpcd = rhx.pxc.getpcd()
    pickle.dump(bgpcd, open("../databackground/tro_bgpcddepthlow1.pkl", "wb"))

    eerot2 = np.dot(rm.rodrigues([0,1,0], 90), eerot)
    armjnts= rhx.movetoposrot(eepos=eepos, eerot=eerot2, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    rhx.pxc.triggerframe()
    bgdepth = rhx.pxc.getdepthimg()
    pickle.dump(bgdepth, open("../databackground/tro_bgdepthlow2.pkl", "wb"))
    bgpcd = rhx.pxc.getpcd()
    pickle.dump(bgpcd, open("../databackground/tro_bgpcddepthlow2.pkl", "wb"))

    eerot3 = np.dot(rm.rodrigues([0,1,0], 180), eerot)
    armjnts= rhx.movetoposrot(eepos=eepos, eerot=eerot3, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    rhx.pxc.triggerframe()
    bgdepth = rhx.pxc.getdepthimg()
    pickle.dump(bgdepth, open("../databackground/tro_bgdepthlow3.pkl", "wb"))
    bgpcd = rhx.pxc.getpcd()
    pickle.dump(bgpcd, open("../databackground/tro_bgpcddepthlow3.pkl", "wb"))

    eerot4 = np.dot(rm.rodrigues([0,1,0], 270), eerot)
    armjnts= rhx.movetoposrot(eepos=eepos, eerot=eerot4, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    rhx.pxc.triggerframe()
    bgdepth = rhx.pxc.getdepthimg()
    pickle.dump(bgdepth, open("../databackground/tro_bgdepthlow4.pkl", "wb"))
    bgpcd = rhx.pxc.getpcd()
    pickle.dump(bgpcd, open("../databackground/tro_bgpcddepthlow4.pkl", "wb"))