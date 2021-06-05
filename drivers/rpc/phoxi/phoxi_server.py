import grpc
import time
import numpy as np
import drivers.devices.phoxi.phoxicontrol as pctrl
from concurrent import futures
import drivers.rpc.phoxi.phoxi_pb2 as pxmsg
import drivers.rpc.phoxi.phoxi_pb2_grpc as pxrpc


class PhoxiServer(pxrpc.PhoxiServicer):

    def __unpackarraydata(self, dobj):
        h = dobj.width
        w = dobj.height
        ch = dobj.channel
        return np.frombuffer(dobj.image, (h,w,ch))

    def initialize(self, pctrlinstance):
        """

        :param pctrlinstance: an instancde of driver.phoxi.phoxicontrol.pyd(PhoxiControl class)
        :return:
        """

        self.__hasframe = False
        self.__pcins = pctrlinstance

    def triggerframe(self, request, context):
        self.__pcins.captureframe()
        self.__width = self.__pcins.getframewidth()
        self.__height = self.__pcins.getframeheight()
        self.__hasframe = True

        return pxmsg.Empty()

    def gettextureimg(self, request, context):
        """
        get texture image as an array

        :return: a Height*Width*1 np array,
        author: weiwei
        date: 20180207
        """

        if self.__hasframe:
            textureraw = self.__pcins.gettexture()
            texturearray = np.array(textureraw).reshape((self.__height, self.__width))
            return pxmsg.CamImg(width=self.__width, height=self.__height, channel=1, image=np.ndarray.tobytes(texturearray))
        else:
            raise ValueError("Trigger a frame first!")

    def getdepthimg(self, request, context):
        """
        get depth image as an array

        :return: a depthHeight*depthWidth*1 np array
        author: weiwei
        date: 20180207
        """

        depthmapraw = self.__pcins.getdepthmap()
        deptharray = np.array(depthmapraw).reshape((self.__height, self.__width))
        return pxmsg.CamImg(width=self.__width, height=self.__height, channel=1, image=np.ndarray.tobytes(deptharray))

    def getpcd(self, request, context):
        """
        get the full poind cloud of a new frame as an array

        :param mat_tw yaml string storing mat_tw
        :return: np.array point cloud n-by-3
        author: weiwei
        date: 20181121
        """

        pointcloudraw = self.__pcins.getpointcloud()
        pcdarray = np.array(pointcloudraw).reshape((-1,3))
        return pxmsg.PointCloud(points=np.ndarray.tobytes(pcdarray))

    def getnormals(self, request, context):
        """
        get the normals of each pixel
        normals share the same datastructure as pointcloud
        the return value is therefore pointcloud

        :param request:
        :param context:
        :return:
        """

        normalsraw = self.__pcins.getnormals()
        normalsarray = np.array(normalsraw).reshape((-1,3))
        return pxmsg.PointCloud(points=np.ndarray.tobytes(normalsarray))

def serve(serialno = "2019-09-051-LC3", host = "127.0.0.1:18300"):
    portno = 65499
    resolution = "low"
    pcins = pctrl.PhoxiControl(serialno, portno, resolution)

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    pxserver = PhoxiServer()
    pxserver.initialize(pcins)
    pxrpc.add_PhoxiServicer_to_server(pxserver, server)
    server.add_insecure_port(host)
    server.start()
    print("The Phoxi server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve(serialno = "2019-04-009-LC3", host = "127.0.0.1:18300")