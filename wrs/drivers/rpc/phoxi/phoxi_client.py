import grpc
import numpy as np
import copy
from wrs import drivers as pxmsg, drivers as pxrpc


class PhxClient(object):

    def __init__(self, host="localhost:18300"):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel(host, options=options)
        self.stub = pxrpc.PhoxiStub(channel)

    def _unpackarraydata(self, dobj):
        h = dobj.height
        w = dobj.width
        ch = dobj.channel
        return copy.deepcopy(np.frombuffer(dobj.image).reshape((h, w, ch)))

    def triggerframe(self):
        self.stub.triggerframe(pxmsg.Empty())

    def gettextureimg(self):
        """
        get gray image as an array

        :return: a textureHeight*textureWidth*1 np array
        author: weiwei
        date: 20191202
        """

        txtreimg = self.stub.gettextureimg(pxmsg.Empty())
        txtrenparray = self._unpackarraydata(txtreimg)
        maxvalue = np.amax(txtrenparray)
        txtrenparray = txtrenparray / maxvalue * 255
        return txtrenparray.astype(np.uint8)

    def getdepthimg(self):
        """
        get depth image as an array

        :return: a depth array array (float32)

        author: weiwei
        date: 20191202
        """

        depthimg = self.stub.getdepthimg(pxmsg.Empty())
        depthnparray = self._unpackarraydata(depthimg)
        depthnparray_float32 = copy.deepcopy(depthnparray)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthnparray, alpha=0.08), cv2.COLORMAP_JET)
        # convert float32 deptharray to unit8
        # maxdepth = np.max(depthnparray)
        # mindepth = np.min(depthnparray[depthnparray != 0])
        # depthnparray[depthnparray!=0] = (depthnparray[depthnparray!=0]-mindepth)/maxdepth*200+55
        # depthnparray = depthnparray.astype(dtype= np.uint8)
        # depthnparray_scaled = copy.deepcopy(depthnparray)
        return depthnparray_float32

    def getpcd(self):
        """
        get the full poind cloud of a new frame as an array

        :return: np.array point cloud n-by-3
        author: weiwei
        date: 20191202
        """

        pcd = self.stub.getpcd(pxmsg.Empty())
        return np.frombuffer(pcd.points).reshape((-1, 3))

    def getnormals(self):
        """
        get the the normals of the pointcloudas an array

        :return: np.array n-by-3
        author: weiwei
        date: 20191208
        """

        nrmls = self.stub.getnormals(pxmsg.Empty())
        return np.frombuffer(nrmls.points).reshape((-1, 3))

    def cvtdepth(self, darr_float32):
        """
        convert float32 deptharray to unit8

        :param darr_float32:
        :return:

        author: weiwei
        date: 20191228
        """

        depthnparray_scaled = copy.deepcopy(darr_float32)
        maxdepth = np.max(darr_float32)
        mindepth = np.min(darr_float32[darr_float32 != 0])
        depthnparray_scaled[depthnparray_scaled != 0] = (darr_float32[darr_float32 != 0] - mindepth) / (maxdepth - mindepth) * 200 + 25
        depthnparray_scaled = depthnparray_scaled.astype(dtype=np.uint8)

        return depthnparray_scaled

    def getrgbtextureimg(self):
        rgbtxtimg = self.stub.getrgbtextureimg(pxmsg.Empty())
        rgbtxtnparray = self._unpackarraydata(rgbtxtimg)

        return rgbtxtnparray.astype(np.uint8)[:,:,::-1]

if __name__ == "__main__":
    import robotconn.rpc.phoxi.phoxi_client as pclt
    import pandaplotutils.pandactrl as pc

    pxc = pclt.PhxClient(host = "192.168.125.100:18300")
    #
    # while True:
    #     pxc.triggerframe()
    #     clrmap, deptharray = pxc.getdepthimg()
    #     cv2.imshow("test", clrmap)
    #     cv2.waitKey(1)


    pcdcenter=[0,0,1500]
    base = pc.World(camp=[0,0,-5000], lookatpos=pcdcenter)

    pcldnp = [None]
    def update(pxc, pcldnp, task):
        if pcldnp[0] is not None:
            pcldnp[0].detachNode()
        pxc.triggerframe()
        pcd = pxc.getpcd()
        normalsnp = pxc.getnormals()
        colorsnp = np.ones((normalsnp.shape[0], normalsnp.shape[1]+1))
        colorsnp[:,:3] = normalsnp
        pcldnp[0] = base.pg.genpointcloudnp(pcd, colors=colorsnp, pntsize=3)
        pcldnp[0].reparentTo(base.render)
        return task.done

    taskMgr.doMethodLater(0.05, update, "update", extraArgs=[pxc, pcldnp],
                      appendTask=True)

    base.run()