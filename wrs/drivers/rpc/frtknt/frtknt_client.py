import grpc
import numpy as np
import copy
from wrs import drivers as fkmsg, drivers as fkrpc

class FrtKnt(object):

    def __init__(self, host = "localhost:18300"):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel(host, options=options)
        self.stub = fkrpc.KntStub(channel)

    def __unpackarraydata(self, dobj):
        h = dobj.width
        w = dobj.height
        ch = dobj.channel
        return copy.deepcopy(np.frombuffer(dobj.image, dtype=np.uint8).reshape((w,h,ch)))

    def getrgbimg(self):
        """
        get color image as an array

        :return: a colorHeight*colorWidth*4 np array, the second and third channels are repeated
        author: weiwei
        date: 20180207
        """

        rgbimg = self.stub.getrgbimg(fkmsg.Empty())
        return self.__unpackarraydata(rgbimg)

    def getdepthimg(self):
        """
        get depth image as an array

        :return: a depthHeight*depthWidth*3 np array, the second and third channels are repeated
        author: weiwei
        date: 20180207
        """

        depthimg = self.stub.getdepthimg(fkmsg.Empty())
        return self.__unpackarraydata(depthimg)

    def getpcd(self):
        """
        get the full poind cloud of a new frame as an array

        :param mat_kw 4x4 nparray
        :return: np.array point cloud n-by-3
        author: weiwei
        date: 20181121
        """

        pcd = self.stub.getpcd(fkmsg.Empty())
        return np.frombuffer(pcd.points, dtype=np.int16).reshape((-1,3))

    def getpartialpcd(self, dframe, width, height):
        """
        get partial poind cloud using the given picklerawdframe, width, height in a depth img

        :param dframe return value of getdepthraw
        :param width, height
        :param picklemat_tw pickle string storing mat_tw

        author: weiwei
        date: 20181121
        """

        widthpairmsg = fkmsg.Pair(data0 = width[0], data1 = width[1])
        heightpairmsg = fkmsg.Pair(data0 = height[0], data1 = height[1])
        h, w, ch = dframe.shape
        dframedata = fkmsg.CamImg(width=w, height=h, channel=ch, image=np.ndarray.tobytes(dframe))
        dframemsg = fkmsg.PartialPcdPara(data = dframedata, width = widthpairmsg,
                                         height = heightpairmsg)
        dobj = self.stub.getpartialpcd(dframemsg)
        return np.frombuffer(dobj.points, dtype=np.int16).reshape((-1,3))

    def mapColorPointToCameraSpace(self, pt):
        """
        convert color space  , to depth space point

        :param pt: [p0, p1] or nparray([p0, p1])
        :return:
        author: weiwei
        date: 20181121
        """

        ptpairmsg = fkmsg.Pair(data0 = pt[0], data1 = pt[1])
        dobj = self.stub.mapColorPointToCameraSpace(ptpairmsg)
        return np.frombuffer(dobj.points, dtype=np.int16).reshape((-1,3))

if __name__ == "__main__":
    import robotconn.rpc.frtknt.frtknt_client as fc
    import cv2

    frk = fc.FrtKnt(host = "10.2.0.60:183001")

    while True:
        img = frk.getdepthimg()
        cv2.imshow("test", img)
        cv2.waitKey(1)


    pcdcenter=[0,0,1500]
    base = pc.World(camp=[0,0,-5000], lookatp=pcdcenter, w=1024, h=768)

    pcldnp = [None]
    def update(frk, pcldnp, task):
        if pcldnp[0] is not None:
            pcldnp[0].detachNode()
        pcd = frk.getpcd()
        pcldnp[0] = base.pg.genpointcloudnp(pcd)
        pcldnp[0].reparentTo(base.render)
        return task.done

    taskMgr.doMethodLater(0.05, update, "update", extraArgs=[frk, pcldnp],
                      appendTask=True)

    base.run()