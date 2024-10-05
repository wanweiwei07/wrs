import grpc
from . import extcam_pb2 as ecmsg
from . import extcam_pb2_grpc as ecrpc
import numpy as np
import cv2

class ExtCam(object):

    def __init__(self, host = "localhost:18300"):
        options = [('grpc.max_receive_message_length', 10 * 3840 * 2160)]
        channel = grpc.insecure_channel(host, options=options)
        self.stub = ecrpc.CamStub(channel)

    def getimg(self):
        message_image = self.stub.getimg(ecmsg.Empty())
        w = message_image.width
        h = message_image.height
        nch = message_image.channel
        imgbytes = message_image.image
        re_img = np.frombuffer(imgbytes, dtype=np.uint8)
        re_img = np.reshape(re_img, (h, w, nch))
        return re_img

if __name__=="__main__":
    ecc = ExtCam(host = "192.168.125.100:18301")
    imgx = ecc.getimg()
    cv2.imshow("name", imgx)
    cv2.waitKey(0)