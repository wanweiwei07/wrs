import grpc
import yaml
import drivers.rpc.hndcam.hndcam_pb2 as hcmsg
import drivers.rpc.hndcam.hndcam_pb2_grpc as hcrpc
import numpy as np
import cv2

class HndCam(object):

    def __init__(self, host = "localhost:18300"):
        self.__oldyaml = True
        if int(yaml.__version__[0]) >= 5:
            self.__oldyaml = False
        channel = grpc.insecure_channel(host)
        self.stub = hcrpc.CamStub(channel)

    def getrc0img(self):
        message_image = self.stub.getrc0img(hcmsg.Empty())
        w = message_image.width
        h = message_image.height
        nch = message_image.channel
        imgbytes = message_image.image
        re_img = np.frombuffer(imgbytes, dtype=np.uint8)
        re_img = np.reshape(re_img, (h, w, nch))
        return re_img

    def getrc1img(self):
        message_image = self.stub.getrc1img(hcmsg.Empty())
        w = message_image.width
        h = message_image.height
        nch = message_image.channel
        imgbytes = message_image.image
        re_img = np.frombuffer(imgbytes, dtype=np.uint8)
        re_img = np.reshape(re_img, (h, w, nch))
        return re_img

    def getlc0img(self):
        message_image = self.stub.getlc0img(hcmsg.Empty())
        w = message_image.width
        h = message_image.height
        nch = message_image.channel
        imgbytes = message_image.image
        re_img = np.frombuffer(imgbytes, dtype=np.uint8)
        re_img = np.reshape(re_img, (h, w, nch))
        return re_img

    def getlc1img(self):
        message_image = self.stub.getlc1img(hcmsg.Empty())
        w = message_image.width
        h = message_image.height
        nch = message_image.channel
        imgbytes = message_image.image
        re_img = np.frombuffer(imgbytes, dtype=np.uint8)
        re_img = np.reshape(re_img, (h, w, nch))
        return re_img

if __name__=="__main__":
    hcc = HndCam(host = "localhost:18300")
    imgx = hcc.getrc0img()
    cv2.imshow("name", imgx)
    cv2.waitKey(0)