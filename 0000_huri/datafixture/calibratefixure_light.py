import pickle
import copy
import utiltools.thirdparty.o3dhelper as o3dh
import utiltools.robotmath as rm
import utiltools.thirdparty.p3dhelper as p3dh
import os


class CalibrateFixture(object):

    def __init__(self, directory=None):
        self.__directory = directory
        if directory is None:
            self.bgdepth = pickle.load(open("./databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open("./databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open("./datacalibration/calibmat.pkl", "rb"))
            self.tstpcdnp = pickle.load(
                open("./dataobjtemplate/tubestand_light_templatepcd.pkl", "rb"))  # tstpcd, tube stand template
            self.tubestandcm = cm.CollisionModel("./objects/tubestand_light.stl")
            self.tubebigcm = cm.CollisionModel("./objects/tubebig_capped.stl", type="cylinder", ex_radius=0)
            self.tubesmallcm = cm.CollisionModel("./objects/tubesmall_capped.stl", type="cylinder", ex_radius=0)
        else:
            self.bgdepth = pickle.load(open(directory + "/databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open(directory + "/databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open(directory + "/datacalibration/calibmat.pkl", "rb"))
            self.tstpcdnp = pickle.load(
                open(directory + "/dataobjtemplate/tubestand_light_templatepcd.pkl", "rb"))  # tstpcd, tube stand template
            self.tubestandcm = cm.CollisionModel(directory + "/objects/tubestand_light.stl")
            self.tubebigcm = cm.CollisionModel(directory + "/objects/tubebig_capped.stl", type="cylinder", ex_radius=0)
            self.tubesmallcm = cm.CollisionModel(directory + "/objects/tubesmall_capped.stl", type="cylinder", ex_radius=0)

    def findtubestand_matchonobb(self, tgtpcdnp, toggledebug=False):
        """
        match self.tstpcd from tgtpcdnp
        using the initilization by findtubestand_obb

        :param tgtpcdnp:
        :param toggledebug:
        :return:

        author: weiwei
        date:20191229osaka
        """

        # toggle the following command to crop the point cloud
        # tgtpcdnp = tgtpcdnp[np.logical_and(tgtpcdnp[:,2]>40, tgtpcdnp[:,2]<60)]

        inithomomat = self.findtubestand_obb(tgtpcdnp, toggledebug)
        inlinnerrmse, homomat = o3dh.registration_icp_ptpt(self.tstpcdnp, tgtpcdnp, inithomomat, maxcorrdist=5,
                                                           toggledebug=toggledebug)
        inithomomatflipped = copy.deepcopy(inithomomat)
        inithomomatflipped[:3, 0] = -inithomomatflipped[:3, 0]
        inithomomatflipped[:3, 1] = -inithomomatflipped[:3, 1]
        inlinnerrmseflipped, homomatflipped = o3dh.registration_icp_ptpt(self.tstpcdnp, tgtpcdnp, inithomomatflipped,
                                                                         maxcorrdist=5, toggledebug=toggledebug)
        print(inlinnerrmse, inlinnerrmseflipped)
        if inlinnerrmseflipped < inlinnerrmse:
            homomat = homomatflipped
        return copy.deepcopy(homomat)

    def findtubestand_match(self, tgtpcdnp, toggledebug=False):
        """
        match self.tstpcd from tgtpcdnp
        NOTE: tgtpcdnp must be in global frame, use getglobalpcd to convert if local

        :param tgtpcdnp:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        _, homomat = o3dh.registration_ptpln(self.tstpcdnp, tgtpcdnp, downsampling_voxelsize=5,
                                             toggledebug=toggledebug)

        return copy.deepcopy(homomat)

    def findtubestand_obb(self, tgtpcdnp, toggledebug=False):
        """
        match self.tstpcd from tgtpcdnp
        NOTE: tgtpcdnp must be in global frame, use getglobalpcd to convert if local

        :param tgtpcdnp:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        tgtpcdnp = o3dh.remove_outlier(tgtpcdnp, nb_points=20, radius=10)

        tgtpcdnp2d = tgtpcdnp[:, :2]  # TODO clip using sensor z
        ca = np.cov(tgtpcdnp2d, y=None, rowvar=0, bias=1)
        v, vect = np.linalg.eig(ca)
        tvect = np.transpose(vect)

        # use the inverse of the eigenvectors as a rotation matrix and
        # rotate the points so they align with the x and y axes
        ar = np.dot(tgtpcdnp2d, np.linalg.inv(tvect))
        # get the minimum and maximum x and y
        mina = np.min(ar, axis=0)
        maxa = np.max(ar, axis=0)
        diff = (maxa - mina) * 0.5
        # the center is just half way between the min and max xy
        center = mina + diff
        # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
        corners = np.array([center + [-diff[0], -diff[1]], center + [diff[0], -diff[1]], center + [diff[0], diff[1]],
                            center + [-diff[0], diff[1]], center + [-diff[0], -diff[1]]])
        # use the the eigenvectors as a rotation matrix and
        # rotate the corners and the centerback
        corners = np.dot(corners, tvect)
        center = np.dot(center, tvect)

        if toggledebug:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            ax.scatter(tgtpcdnp2d[:, 0], tgtpcdnp2d[:, 1])
            ax.scatter([center[0]], [center[1]])
            ax.plot(corners[:, 0], corners[:, 1], '-')
            plt.axis('equal')
            plt.show()

        axind = np.argsort(v)
        homomat = np.eye(4)
        homomat[:3, axind[0]] = np.array([vect[0, 0], vect[1, 0], 0])
        homomat[:3, axind[1]] = np.array([vect[0, 1], vect[1, 1], 0])
        homomat[:3, 2] = np.array([0, 0, 1])
        if np.cross(homomat[:3, 0], homomat[:3, 1])[2] < -.5:
            homomat[:3, 1] = -homomat[:3, 1]
        homomat[:3, 3] = np.array([center[0], center[1], -15])
        return homomat

    def capturecorrectedpcd(self, pxc, ncapturetimes=1):
        """
        capture a poind cloud and transform it from its sensor frame to global frame

        :param pcdnp:
        :return:

        author: weiwei
        date: 20200108
        """

        objpcdmerged = None
        for i in range(ncapturetimes):
            pxc.triggerframe()
            fgdepth = pxc.getdepthimg()
            fgpcd = pxc.getpcd()

            substracteddepth = self.bgdepth
            substracteddepth = self.bgdepth - fgdepth
            substracteddepth = substracteddepth.clip(50, 70)
            substracteddepth[substracteddepth == 50] = 0
            substracteddepth[substracteddepth == 70] = 0

            tempdepth = substracteddepth.flatten()
            objpcd = fgpcd[np.nonzero(tempdepth)]
            objpcd = self.getcorrectedpcd(objpcd)
            if objpcdmerged is None:
                objpcdmerged = objpcd
            else:
                objpcdmerged = np.vstack((objpcdmerged, objpcd))
        objpcdmerged = objpcdmerged[objpcdmerged[:,0]>200]

        return objpcdmerged

    def getcorrectedpcd(self, pcdarray):
        """
        convert a poind cloud from its sensor frame to global frame

        :param pcdarray:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        return rm.homotransformpointarray(self.sensorhomomat, pcdarray)

    def gentubestand(self, homomat):
        """

        :param homomat:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        tubestandcm = copy.deepcopy(self.tubestandcm)
        tubestandcm.set_homomat(homomat)
        tubestandcm.setColor(0, .5, .7, 1)

        return tubestandcm


if __name__ == '__main__':
    import robothelper
    import numpy as np
    import environment.collisionmodel as cm

    yhx = robothelper.RobotHelperX(usereal=False, startworld=True)
    cf = CalibrateFixture(directory=yhx.path)

    bgdepth = pickle.load(open(yhx.path + "/databackground/bgdepth.pkl", "rb"))
    bgpcd = pickle.load(open(yhx.path + "/databackground/bgpcd.pkl", "rb"))

    objpcd = cf.capturecorrectedpcd(yhx.pxc, ncapturetimes=1)
    pcdnp = p3dh.genpointcloudnodepath(objpcd, pntsize=5)
    pcdnp.reparentTo(yhx.base.render)
    yhx.base.run()

    homomat = cf.findtubestand_matchonobb(objpcd, toggledebug=False)
    tbscm = cf.gentubestand(homomat=homomat)
    print(homomat)
    pickle.dump(homomat, open(os.path.join(yhx.path, "datafixture", "rightfixture_light_homomat1.pkl"), "wb"))
    tbscm.reparentTo(yhx.base.render)

    yhx.base.run()
