import os
import cv2
import time
import glob
import numpy as np
import cv2.aruco as aruco
import yaml
import math
import utiltools.robotmath as rm

def genworldpoints(nrow, ncolumn, markersize):
    """
    get the realworld positions of the chessmarkers
    z is 0
    upperleft is the origin

    :param nrow:
    :param ncolumn:
    :param markersize:
    :return:

    author: weiwei
    date: 20190420
    """

    worldpoints = np.zeros((nrow*ncolumn, 3), np.float32)
    worldpoints[:, :2] = np.mgrid[:nrow, :ncolumn].T.reshape(-1, 2)*markersize
    return worldpoints

def captureimgbytime(camid=0, type="png", timeinterval=1):
    """

    :param camid:
    :param type:
    :param timeinterval: seconds
    :return:
    """

    camera = cv2.VideoCapture(camid)
    imgid = 0
    foldername = "./camimgs" + str(imgid) + "/"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    else:
        files = glob.glob(foldername+"*")
        for f in files:
            os.remove(f)
    while True:
        return_value, image = camera.read()
        cv2.imwrite(foldername+'opencv' + str(imgid) + '.' + type, image)
        print("Saving an image...ID"+str(imgid))
        cv2.imshow('The images...', image)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        imgid+=1
        time.sleep(timeinterval)

def captureimgbytimemulticam(camids=[0,2,4], type="png", timeinterval=1):
    """

    :param camid:
    :param type:
    :param timeinterval: seconds
    :return:
    """

    savepaths = []
    cameras = []
    windows = []
    for camid in camids:
        foldername = "./camimgs"+str(camid)+"/"
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        else:
            files = glob.glob(foldername+"*")
            for f in files:
                os.remove(f)
        savepaths.append(foldername)
        cameras.append(cv2.VideoCapture(camid))
        windows.append('cam'+str(camid))
    imgid = 0
    while True:
        returnvalues = []
        images = []
        for id, camera in enumerate(cameras):
            returnvalue, image = camera.read()
            images.append(image)
            returnvalues.append(returnvalue)
        for id, camera in enumerate(cameras):
            cv2.imwrite(savepaths[id]+'opencv' + str(imgid) + '.' + type, images[id])
            cv2.imshow(windows[id], images[id])
            cv2.moveWindow(windows[id], 450+id*700, 200)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                return
        imgid+=1
        time.sleep(timeinterval)

def captureimgbyspacemulticam(camids=[0,2,4], width=640, height=480, type="png"):
    """

    :param camid:
    :param type:
    :return:
    """

    savepaths = []
    cameras = []
    windows = []
    for camid in camids:
        foldername = "./camimgs"+str(camid)+"/"
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        else:
            files = glob.glob(foldername+"*")
            for f in files:
                os.remove(f)
        savepaths.append(foldername)
        cameras.append(cv2.VideoCapture(camid))
        cameras[-1].set(cv2.CAP_PROP_FPS, 30)
        cameras[-1].set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cameras[-1].set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        windows.append('cam'+str(camid))
    imgid = 0
    while True:
        returnvalues = []
        images = []
        for id, camera in enumerate(cameras):
            returnvalue, image = camera.read()
            images.append(image)
            returnvalues.append(returnvalue)
        for id, camera in enumerate(cameras):
            cv2.imshow(windows[id], images[id])
            cv2.moveWindow(windows[id], 0+id*1300, 200)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            return
        if k%256 == 32:
            # ESC pressed
            for id, camera in enumerate(cameras):
                cv2.imwrite(savepaths[id]+'opencv' + str(imgid) + '.' + type, images[id])
            imgid+=1
            # Space bar pressed
            print("Saving an image...ID"+str(imgid))
        time.sleep(.5)

def captureimgbychessdetect(nrow, ncolumn, camid=0, type="png"):
    """

    :param camid:
    :param nrow: nrow of checker board
    :param ncolumn: ncolumn of checker board
    :param savepath:
    :return:
    """

    camera = cv2.VideoCapture(camid)
    imgid = 0
    foldername = "./camimgs" + str(imgid) + "/"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    else:
        files = glob.glob(foldername+"*")
        for f in files:
            os.remove(f)
    lastcaptime = time.time()
    while True:
        newcaptime = time.time()
        return_value, image = camera.read()
        cv2.imshow('The images...', image)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        ret, corners = cv2.findChessboardCorners(image, (ncolumn, nrow))
        if ret and (len(corners) == nrow*ncolumn) and (newcaptime-lastcaptime > 1):
            cv2.imwrite(foldername+'opencv' + str(imgid) + type, image)
            print("Saving an image...ID"+str(imgid))
            lastcaptime = newcaptime
            imgid+=1

def estimatemarkerpose(imglist, calibrationdata, arucomarkersize=50, aruco_dict=aruco.DICT_4X4_250):
    """

    :param imglist:
    :param calibrationdata: could be a yamlfilpath or a list [mtx, dist, rvecs, tvecs]
    :param arucomarkersize:
    :param aruco_dict:
    :return:
    """

    if isinstance(calibrationdata, str):
        mtx, dist, rvecs, tvecs, _ = yaml.load(open(calibrationdata, 'r'), Loader=yaml.UnsafeLoader)
    else:
        mtx, dist, rvecs, tvecs = calibrationdata

    aruco_dict = aruco.Dictionary_get(aruco_dict)
    parameters = aruco.DetectorParameters_create()

    poslist = []
    rotlist = []
    for img in imglist:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        if ids is not None:
            aruco.drawDetectedMarkers(img, corners, borderColor=[255,255,0])
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, arucomarkersize, mtx, dist)
            aruco.drawAxis(img, mtx, dist, rvecs[0], tvecs[0]/1000.0, 0.1)
            rot = cv2.Rodrigues(rvecs[0])[0]
            pos = tvecs[0][0].ravel()
            rotlist.append(rot)
            poslist.append(pos)
    posavg = rm.posvec_average(poslist)
    rotavg = rm.rotmat_average(rotlist)

    return [posavg, rotavg]

def estimatecorners(img):
    """

    :return:
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    if dst is not None:
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv2.imshow('dst', img)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    return []

def computefov(mtx, imgwidth, imgheight):
    """
    This function is implemented by referring to
    line 1817 of https://github.com/opencv/opencv/blob/2.4/modules/calib3d/src/calibration.cpp#L1778
    Different from the one in opencv,
    this function does not need the aperture size, which might be unavailable for some webcams

    Usage: only use fov_h when setting up a new camera in panda3d,
    the fov_v will be automatically recomputed considering the width and height of an image
    and assuming box pixels

    :return: fov_h and fov_v, horizontal and vertical fov
    """

    fov_h = math.degrees(2*math.atan(imgwidth/(2*mtx[0,0])))
    fov_v = math.degrees(2*math.atan(imgheight/(2*mtx[1,1])))
    return [fov_h, fov_v]

if __name__=='__main__':
    # makechessboard(7,5,markersize=40)
    # worldpoints = genworldpoints(8,6, 25)
    # print(worldpoints)
    # captureimgbydetect(8,6)
    # captureimgbytimemulticam()

    import robotconn.rpc.frtknt.frtknt_client as fkc
    fk = fkc.FrtKnt(host="192.168.125.60:18300")
    img = fk.getrgbimg()
    estimatecorners(img)