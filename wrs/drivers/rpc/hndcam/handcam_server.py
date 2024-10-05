import grpc
import time
from concurrent import futures
import numpy as np
import wrs.drivers.rpc.hndcam.cameras as cameras
from wrs import drivers as hcmsg, drivers as hcrpc

class HndCamServer(hcrpc.CamServicer):

    def __init__(self):
        self.hc = cameras.HndCam()

    def getrc0img(self, request, context):
        """

        Inherited from the auto-generated handcam_pb2_grpc

        :param request:
        :param context:
        :return:

        author: weiwei
        date: 20190416, 20190609osaka
        """

        frame = self.hc.getrc0img()
        h, w, nch = frame.shape
        fmbytes = np.ndarray.tobytes(frame)
        return hcmsg.CamImg(width=w, height=h, channel=nch, image=fmbytes)

    def getrc1img(self, request, context):
        """

        Inherited from the auto-generated handcam_pb2_grpc

        :param request:
        :param context:
        :return:

        author: weiwei
        date: 20190416
        """

        frame = self.hc.getrc1img()
        h, w, nch = frame.shape
        fmbytes = np.ndarray.tobytes(frame)
        return hcmsg.CamImg(width=w, height=h, channel=nch, image=fmbytes)

    def getlc0img(self, request, context):
        """

        Inherited from the auto-generated handcam_pb2_grpc

        :param request:
        :param context:
        :return:

        author: weiwei
        date: 20190416
        """

        frame = self.hc.getlc0img()
        h, w, nch = frame.shape
        fmbytes = np.ndarray.tobytes(frame)
        return hcmsg.CamImg(width=w, height=h, channel=nch, image=fmbytes)

    def getlc1img(self, request, context):
        """

        Inherited from the auto-generated handcam_pb2_grpc

        :param request:
        :param context:
        :return:

        author: weiwei
        date: 20190416
        """

        frame = self.hc.getlc1img()
        h, w, nch = frame.shape
        fmbytes = np.ndarray.tobytes(frame)
        return hcmsg.CamImg(width=w, height=h, channel=nch, image=fmbytes)

def serve():
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hcrpc.add_CamServicer_to_server(HndCamServer(), server)
    server.add_insecure_port('[::]:18300')
    server.start()
    print("The HndCam server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()