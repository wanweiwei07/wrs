import grpc
import time
import numpy as np
from concurrent import futures
from . import cameras
from . import extcam_pb2 as ecmsg
from . import extcam_pb2_grpc as ecrpc

class ExtCamServer(ecrpc.CamServicer):

    def __init__(self):
        self.ec = cameras.ExtCam()

    def getimg(self, request, context):
        """

        Inherited from the auto-generated extcam_pb2_grpc

        :param request:
        :param context:
        :return:

        author: weiwei
        date: 20190416, 20190609, 20191227osaka
        """

        frame = self.ec.getimg()
        h, w, nch = frame.shape
        fmbytes = np.ndarray.tobytes(frame)
        return ecmsg.CamImg(width=w, height=h, channel=nch, image=fmbytes)

def serve(host="127.0.0.1:18300"):
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24

    options = [('grpc.max_message_length', 10 * 3840 * 2160)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    ecrpc.add_CamServicer_to_server(ExtCamServer(), server)
    server.add_insecure_port(host)
    server.start()
    print("The ExtCam server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve(host="192.168.125.100:18301")