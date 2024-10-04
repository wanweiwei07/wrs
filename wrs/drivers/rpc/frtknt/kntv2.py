#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import threading
import time

from wrs._misc import PyKinectRuntime

import ctypes

class KinectV2(PyKinectRuntime.PyKinectRuntime):
    def __init__(self, FrameSourceTypes):
        PyKinectRuntime.PyKinectRuntime.__init__(self, FrameSourceTypes)

        self.__color = None
        self.__color_width = self.color_frame_desc.Width     # 1920
        self.__color_height = self.color_frame_desc.Height   # 1080
        print('color size = (%d, %d)' % (self.__color_width, self.__color_height))

        self.__depth = None
        self.__depth_record = None
        self.__depth_width = self.depth_frame_desc.Width     # 512
        self.__depth_height = self.depth_frame_desc.Height   # 424
        print('depth size = (%d, %d)' % (self.__depth_width, self.__depth_height))

        # self.__infrared = None
        # self.__infrared_width = self.infrared_frame_desc.Width
        # self.__infrared_height = self.infrared_frame_desc.Height
        # print('infrared size = (%d, %d)' % (self.__infrared_width, self.__infrared_height))

        # system timestample (ms), based on lambda
        self.nowTime = lambda: int(round(time.time() * 1000))

        # timestample
        self.color_timestample = None
        self.depth_timestample = None
        # self.infrared_timestample = None

        # match center
        self.__center = None

    @property
    def depthRecord(self):
        # read-only property
        return self.__depth_record

    @property
    def depthWidth(self):
        # read-only property
        return self.__depth_width

    @property
    def depthHeight(self):
        # read-only property
        return self.__depth_height

    @property
    def colorWidth(self):
        # read-only property
        return self.__color_width

    @property
    def colorHeight(self):
        # read-only property
        return self.__color_height

    # @property
    # def infraredWidth(self):
    #     # read-only property
    #     return self.__infrared_width
    #
    # @property
    # def infraredHeight(self):
    #     # read-only property
    #     return self.__infrared_height

    @property
    def circleCenter(self):
        # read-only property
        return self.__center

    def getColorFrame(self):
        # read-only property
        return self.__color

    def getDepthFrame(self):
        # read-only property
        return self.__depth

    # def getInfraredFrame(self):
    #     # read-only property
    #     return self.__infrared

    def mapCameraPointToDepthSpace(self, pt):
        return self._mapper.MapCameraPointToDepthSpace(pt)

    def mapCameraPointToColorSpace(self, pt):
        return self._mapper.MapCameraPointToColorSpace(pt)

    def mapColorPointToCameraSpace(self, pt):
        dframe = self.getDepthFrame()
        pcdptr = ctypes.cast((PyKinectV2._CameraSpacePoint * 1920*1080)(), ctypes.POINTER(PyKinectV2._CameraSpacePoint))
        self._mapper.MapColorFrameToCameraSpace(512*424, dframe.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)), 1920*1080, pcdptr)
        # self._mapper.MapDepthFrameToCameraSpace(npoints, dframe.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)), npoints, pcdptr)
        obj = ctypes.cast(pcdptr, ctypes.POINTER(np.ctypeslib._ctype_ndarray(ctypes.c_float, (1920*1080, 3)))).contents
        pcd = np.array(obj)
        pt[0] = int(pt[0])
        pt[1] = int(pt[1])
        return pcd[(pt[1]-1)*1920+pt[0]]*1000.0

    def getDepthWidthAndHeight(self):
        return (self.__depth_width, self.__depth_height)

    def __getXYZForPixel(self, dframe, px):
        """
        Convert a depth frame pixel to a 3D point

        :param dframe: depthframe from pykinectv2
        :param px: 1-by-2 list
        :return: 1-by-3 list [x, y, z] or None

        author: weibo
        date: 20180110
        """

        u, v = int(px[0]), int(px[1])
        z3d = dframe[v * self.__depth_width + u]

        if z3d <= 0:
            return None

        point3d = self._mapper.MapDepthPointToCameraSpace(
            PyKinectV2._DepthSpacePoint(ctypes.c_float(u), ctypes.c_float(v)), ctypes.c_ushort(z3d))

        return [point3d.x * 1000.0, point3d.y * 1000.0, point3d.z * 1000.0]


    def getPointCloud(self, dframe, width=[0, 512], height=[0, 424], mat_kw=None):
        """
        get the point cloud of the depth frame

        :param dframe:
        :param width:
        :param height:
        :param mat_kw: transform points to world space
        :return: np.array point cloud n-by-3

        author: weiwei
        date: 20181130
        """

        dframecrop = dframe[height[0]:height[1]][width[0]:width[1]]
        npoints = (height[1]-height[0])*(width[1]-width[0])
        pcdptr = ctypes.cast((PyKinectV2._CameraSpacePoint*npoints)(), ctypes.POINTER(PyKinectV2._CameraSpacePoint))
        self._mapper.MapDepthFrameToCameraSpace(npoints, dframecrop.ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)),
                                               npoints, pcdptr)
        obj = ctypes.cast(pcdptr, ctypes.POINTER(np.ctypeslib._ctype_ndarray(ctypes.c_float, (npoints,3)))).contents
        pcd = np.array(obj, dtype=np.float32)
        if mat_kw is None:
            return pcd*1000
        else:
            # TODO transform
                return np.array([])

    def recordDepth(self):
        """
        record a depth frame

        :return:

        author: webo
        data: 201712227
        """
        self.__depth_record = self.__depth

    def runOnce(self):
        """
        call this function once to get the depth, infrared and color

        :return:
        """
        while True:
            if self.has_new_depth_frame():
                self.__depth = self.get_last_depth_frame()
                self.depth_timestample = str(self.nowTime())

            # if self.has_new_infrared_frame():
            #     self.__infrared = self.get_last_infrared_frame()
            #     self.infrared_timestample = str(self.nowTime())

            if self.has_new_color_frame():
                self.__color = self.get_last_color_frame()
                self.color_timestample = str(self.nowTime())

            # --- Limit to 60 frames per second
            self.__clock.tick(60)

        # Close Kinect sensor, close the window and quit.
        print('Close Kinect sensor, close the window and quit.')
        self.close()

    def runForThread(self):
        """
        call this function in a thread

        :return:
        """
        while True:
            strtime = str(self.nowTime())
            if self.has_new_depth_frame():
                self.__depth = self.get_last_depth_frame()
                self.depth_timestample = strtime

            # if self.has_new_infrared_frame():
            #     self.__infrared = self.get_last_infrared_frame()
            #     self.infrared_timestample = strtime

            if self.has_new_color_frame():
                self.__color = self.get_last_color_frame()
                self.color_timestample = strtime
        # Close Kinect sensor, close the window and quit.
        print('Close Kinect sensor, close the window and quit.')
        self.close()

class ThreadKinectCam(threading.Thread):
    def __init__(self, index, create_time, kinect):
        threading.Thread.__init__(self)
        self.index = index
        self.create_time = create_time
        self.kinect = kinect

    def close(self):
        self.kinect.close()

    def run(self):
        self.kinect.runForThread()

if __name__ == "__main__":
    import cv2
    from pykinect2 import PyKinectV2
    from pykinect2 import PyKinectRuntime

    kinect = KinectV2(PyKinectV2.FrameSourceTypes_Color |
                      PyKinectV2.FrameSourceTypes_Depth |
                      PyKinectRuntime.FrameSourceTypes_Infrared)

    threadKinectCam = ThreadKinectCam(2, time.time(), kinect)
    threadKinectCam.start()
    while True:
        if kinect.getColorFrame() is None:
            print("initializing color...")
            continue
        if kinect.getDepthFrame() is None:
            print("initializing depth...")
            continue
        break
    while True:
        clframe = kinect.getColorFrame()
        clb = np.flip(np.array(clframe[0::4]).reshape((kinect.colorHeight, kinect.colorWidth)),1)
        clg = np.flip(np.array(clframe[1::4]).reshape((kinect.colorHeight, kinect.colorWidth)),1)
        clr = np.flip(np.array(clframe[2::4]).reshape((kinect.colorHeight, kinect.colorWidth)),1)
        # cla = np.array(clframe[3::4])
        clframe8bit = np.dstack((clb, clg, clr)).reshape((kinect.colorHeight, kinect.colorWidth, 3))
        img = cv2.merge((clb,clg,clr))
        # img = cv2.resize(img, (int(kinect.colorWidth/3.0), int(kinect.colorHeight/3.0)))
        cv2.imwrite("test.jpg", img)
        aruco = cv2.aruco
        arucodict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        #
        # camcap = cv2.VideoCapture(3)
        # img = camcap.read()[1]
        corners,ids, rejectedImgPoints = aruco.detectMarkers(img, arucodict)
        print(kinect.mapColorPointToCameraSpace([950,974]))
        aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
        cv2.imshow("xx", img)
        cv2.waitKey(10)
        