import os
import pickle

import numpy as np

import config
# import robotcon.rpc.phoxi.phoxi_client as phoxiclient
from wrs import drivers as phoxiclient


class Phoxi(object):
    def __init__(self, host="127.0.0.1:18300"):
        self.client = phoxiclient.PhxClient(host=host)

    def load_phoxicalibmat(self, amat_path=os.path.join(config.ROOT, "camcalib/data/"), f_name="phoxi_calibmat.pkl"):
        amat = pickle.load(open(os.path.join(amat_path, f_name), "rb"))
        return amat

    def getdepthimg(self):
        self.client.triggerframe()
        clrmap, deptharray = self.client.getdepthimg()
        # cv2.imshow("test", clrmap)
        # cv2.waitKey(1)
        return [clrmap, deptharray]

    def showdepthimg(self):
        self.client.triggerframe()
        depthnparray_float32 = self.client.getdepthimg()
        depthnparray_scaled = self.scalefloat32uint8(depthnparray_float32)
        # cv2.imshow("test", clrmap)
        # cv2.waitKey(1)
        return depthnparray_scaled

    def getgrayimg(self):
        self.client.triggerframe()
        return self.client.gettextureimg()

    def getalldata(self):
        self.client.triggerframe()
        grayimg = self.client.gettextureimg()
        depthnparray_float32 = self.client.getdepthimg()
        pcd = self.client.getpcd()
        # normals = self.client.getnormals()

        return [grayimg, depthnparray_float32, pcd]

    def dumpalldata(self, f_name="/phoxi_tempdata.pkl"):
        self.client.triggerframe()
        grayimg = self.client.gettextureimg()
        depthnparray_float32 = self.client.getdepthimg()
        pcd = self.client.getpcd()
        # normals = self.client.getnormals()
        with open(os.path.join(config.ROOT, f_name), 'wb') as f:
            pickle.dump([grayimg, depthnparray_float32, pcd], f)

        return [grayimg, depthnparray_float32, pcd]

    def loadalldata(self, f_name="./phoxi_tempdata.pkl"):
        alldata = pickle.load(open(os.path.join(config.ROOT, f_name), "rb"))
        return alldata

    def scalefloat32uint8(self, nparrayfloat32, difffromzero=55):
        """
        scale an depthimg (float32) to (uint8)

        :param nparrayfloat32:
        :param difffromzero: the difference between 0 and the minimum value in the converted uint8 image
        :return:
        """

        maxdepth = np.max(nparrayfloat32)
        mindepth = np.min(nparrayfloat32[nparrayfloat32 != 0])
        nparrayuint8 = np.array(nparrayfloat32).astype(dtype=np.uint8)
        nparrayuint8[nparrayfloat32 != 0] = (nparrayfloat32[nparrayfloat32 != 0] - mindepth) / maxdepth * (
                255 - difffromzero) + difffromzero
        return nparrayuint8
