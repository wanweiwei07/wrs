import grpc
import time
import numpy as np
from concurrent import futures
import robotconn.rpc.frtknt.kntv2 as kntv2
import robotconn.rpc.frtknt.frtknt_pb2 as fkmsg
import robotconn.rpc.frtknt.frtknt_pb2_grpc as fkrpc
from wrs.drivers import PyKinectV2


class FrtKntServer(fkrpc.KntServicer):

    def __unpackarraydata(self, dobj):
        h = dobj.width
        w = dobj.height
        ch = dobj.channel
        return np.frombuffer(dobj.image, (h,w,ch))

    def initialize(self, kinect):
        self.__kinect = kinect

    def getrgbimg(self, request, context):
        """
        get color image as an array

        :return: a colorHeight*colorWidth*4 np array, the second and third channels are repeated
        author: weiwei
        date: 20180207
        """

        clframe = self.__kinect.getColorFrame()
        clb = np.flip(np.array(clframe[0::4]).reshape((self.__kinect.colorHeight, self.__kinect.colorWidth)),1)
        clg = np.flip(np.array(clframe[1::4]).reshape((self.__kinect.colorHeight, self.__kinect.colorWidth)),1)
        clr = np.flip(np.array(clframe[2::4]).reshape((self.__kinect.colorHeight, self.__kinect.colorWidth)),1)
        channel = 3
        clframe8bit = np.dstack((clb, clg, clr)).reshape((self.__kinect.colorHeight, self.__kinect.colorWidth, channel))
        return fkmsg.CamImg(width=self.__kinect.colorWidth, height=self.__kinect.colorHeight, channel=channel, image=np.ndarray.tobytes(clframe8bit))

    def getdepthimg(self, request, context):
        """
        get depth image as an array

        :return: a depthHeight*depthWidth*3 np array, the second and third channels are repeated
        author: weiwei
        date: 20180207
        """

        dframe = self.__kinect.getDepthFrame()
        df8 = np.uint8(dframe.clip(1, 4000) / 16.)
        channel = 1
        dframe8bit = np.array(df8).reshape((self.__kinect.depthHeight, self.__kinect.depthWidth, channel))
        return fkmsg.CamImg(width=self.__kinect.depthWidth, height=self.__kinect.depthHeight, channel=channel, image=np.ndarray.tobytes(dframe8bit))

    def getpcd(self, request, context):
        """
        get the full poind cloud of a new frame as an array

        :param mat_tw yaml string storing mat_tw
        :return: np.array point cloud n-by-3
        author: weiwei
        date: 20181121
        """

        dframe = self.__kinect.getDepthFrame()
        pcdarray = self.__kinect.getPointCloud(dframe)
        return fkmsg.PointCloud(points=np.ndarray.tobytes(pcdarray.astype(dtype=np.int16)))

    def getpartialpcd(self, request, context):
        """
        get partial poind cloud using the given 8bitdframe, width, height in a depth img

        :param rawdframe yaml string storing raw dframe
        :param width, height
        :param picklemat_tw pickle string storing mat_tw
``````````````        author: weiwei
        date: 20181121
        """

        width = [request.width.data0, request.width.data1]
        height = [request.height.data0, request.width.data1]
        dframe = self.__unpackarraydata(request.data)
        pcdarray = self.__kinect.getPointCloud(dframe, width, height)
        return fkmsg.PointCloud(points=np.ndarray.tobytes(pcdarray.astype(dtype=np.int16)))

    def mapColorPointToCameraSpace(self, request, context):
        """
        convert color space  , to depth space point

        :param pt:
        :return:
        author: weiwei
        date: 20181121
        """

        return fkmsg.PointCloud(points=np.ndarray.tobytes(self.__kinect.mapColorPointToCameraSpace([request.data0, request.data1]).astype(dtype=np.int16)))

def serve(host = "127.0.0.1:18300"):
    kinect = kntv2.KinectV2(PyKinectV2.FrameSourceTypes_Color |
                            PyKinectV2.FrameSourceTypes_Depth)
                            # | PyKinectRuntime.FrameSourceTypes_Infrared)
    threadKinectCam = kntv2.ThreadKinectCam(2, time.time(), kinect)
    threadKinectCam.start()
    while True:
        # if kinect.getInfraredFrame() is None:
        #     print("initializing infrared...")
        #     continue
        if kinect.getColorFrame() is None:
            print("initializing color...")
            continue
        if kinect.getDepthFrame() is None:
            print("initializing depth...")
            continue
        break
    print("Kinect Initialized!")

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    frtserver = FrtKntServer()
    frtserver.initialize(kinect)
    fkrpc.add_KntServicer_to_server(frtserver, server)
    server.add_insecure_port(host)
    server.start()
    print("The Front Kinect server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve(host = "10.2.0.60:183001")