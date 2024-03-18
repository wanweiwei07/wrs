import pickle
import copy
import numpy as np
import environment.collisionmodel as cm
import utiltools.thirdparty.o3dhelper as o3dh
import utiltools.robotmath as rm
import utiltools.thirdparty.p3dhelper as p3dh

class LocatorFixed(object):

    def __init__(self, directory=None, homomatfilename_start="rightfixture_light_homomat1", homomatfilename_goal="rightfixture_light_homomat2"):
        self.__directory = directory
        if directory is None:
            self.bgdepth = pickle.load(open("./databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open("./databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open("./datacalibration/calibmat.pkl", "rb"))
            self.tubestandcm = cm.CollisionModel("./objects/tubestand_light.stl")
            self.tubebigcm = cm.CollisionModel("./objects/tubebig_capped.stl", type="cylinder", ex_radius=2)
            self.tubesmallcm = cm.CollisionModel("./objects/tubesmall_capped.stl", type="cylinder", ex_radius=2)
            self.tubestandhomomat_start = pickle.load(open("./datafixture/"+homomatfilename_start+".pkl", "rb"))
            self.tubestandhomomat_goal = pickle.load(open("./datafixture/"+homomatfilename_goal+".pkl", "rb"))
        else:
            self.bgdepth = pickle.load(open(directory+"/databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open(directory+"/databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open(directory + "/datacalibration/calibmat.pkl", "rb"))
            self.tubestandcm = cm.CollisionModel(directory + "/objects/tubestand_light.stl")
            self.tubebigcm = cm.CollisionModel(directory + "/objects/tubebig_capped.stl", type="cylinder", ex_radius=2)
            self.tubesmallcm = cm.CollisionModel(directory + "/objects/tubesmall_capped.stl", type="cylinder", ex_radius=2)
            self.tubestandhomomat_start = pickle.load(open("./datafixture/"+homomatfilename_start+".pkl", "rb"))
            self.tubestandhomomat_goal = pickle.load(open("./datafixture/"+homomatfilename_goal+".pkl", "rb"))

        # down x, right y
        tubeholecenters = []
        for x in [-36, -18, 0, 18, 36]:
            tubeholecenters.append([])
            for y in [-83.25, -64.75, -46.25, -27.75, -9.25, 9.25, 27.75, 46.25, 64.75, 83.25]:
                tubeholecenters[-1].append([x, y])
        self.tubeholecenters = np.array(tubeholecenters)
        self.tubeholesize = np.array([17, 16.5])
        self.tubestandsize = np.array([97, 191])
        # initialize the registered tubes, a dictionary with the template of each tube end_type in a list (multiple values allowed)
        self.registeredtubetemps = {1:[], 2:[]}

    def _crop_pcd_overahole(self, tgtpcd_intsframe, holecenter_x, holecenter_y, crop_ratio = .9, crop_height = 70):
        """

        crop the point cloud over a hole in the tubestand frame

        :param tgtpcd_intsframe:
        :param holecenter_x, holecenter_y:
        :param crop_ratio:
        :param crop_height:
        :return:

        author: weiwei
        date: 20200318
        """

        # squeeze the hole size by half, make it a bit smaller than a half
        tmppcd = tgtpcd_intsframe[tgtpcd_intsframe[:, 0] < (holecenter_x + self.tubeholesize[0]*crop_ratio/2)]
        tmppcd = tmppcd[tmppcd[:, 0] > (holecenter_x - self.tubeholesize[0]*crop_ratio/2)]
        tmppcd = tmppcd[tmppcd[:, 1] < (holecenter_y + self.tubeholesize[1]*crop_ratio/2)]
        tmppcd = tmppcd[tmppcd[:, 1] > (holecenter_y - self.tubeholesize[1]*crop_ratio/2)]
        tmppcd = tmppcd[tmppcd[:, 2] > crop_height]

        return tmppcd

    def findtubes(self, tubestand_homomat, tgtpcdnp, toggledebug=False):
        """

        :param tgtpcdnp:
        :return:

        author: weiwei
        date: 20200317
        """

        elearray = np.zeros((5, 10))
        eleconfidencearray = np.zeros((5, 10))

        tgtpcdnp = o3dh.remove_outlier(tgtpcdnp, downsampling_voxelsize=None, nb_points=90, radius=5)
        # transform back to the local frame of the tubestand
        tgtpcdnp_normalized = rm.homotransformpointarray(rm.homoinverse(tubestand_homomat), tgtpcdnp)
        if toggledebug:
            cm.CollisionModel(tgtpcdnp_normalized).reparentTo(base.render)
            if self.__directory is None:
                tscm2 = cm.CollisionModel("./objects/tubestand.stl")
            else:
                tscm2 = cm.CollisionModel(self.__directory + "/objects/tubestand.stl")
            tscm2.reparentTo(base.render)
        for i in range(5):
            for j in range(10):
                holepos = self.tubeholecenters[i][j]
                tmppcd = self._crop_pcd_overahole(tgtpcdnp_normalized, holepos[0], holepos[1])
                if len(tmppcd) > 50:
                    if toggledebug:
                        print("------more than 50 raw points, start a new test------")
                    tmppcdover100 = tmppcd[tmppcd[:, 2] > 100]
                    tmppcdbelow90 = tmppcd[tmppcd[:, 2] < 90]
                    tmppcdlist = [tmppcdover100, tmppcdbelow90]
                    if toggledebug:
                        print("rotate around...")
                    rejflaglist = [False, False]
                    allminstdlist = [[], []]
                    newtmppcdlist = [None, None]
                    minstdlist = [None, None]
                    for k in range(2):
                        if toggledebug:
                            print("checking over 100 and below 90, now: ", j)
                        if len(tmppcdlist[k]) < 10:
                            rejflaglist[k] = True
                            continue
                        for angle in np.linspace(0, 180, 10):
                            tmphomomat = np.eye(4)
                            tmphomomat[:3, :3] = rm.rodrigues(tubestand_homomat[:3, 2], angle)
                            newtmppcdlist[k] = rm.homotransformpointarray(tmphomomat, tmppcdlist[k])
                            minstdlist[k] = np.min(np.std(newtmppcdlist[k][:, :2], axis=0))
                            if toggledebug:
                                print(minstdlist[k])
                            allminstdlist[k].append(minstdlist[k])
                            if minstdlist[k] < 1.5:
                                rejflaglist[k] = True
                        if toggledebug:
                            print("rotate round done")
                            print("minstd ", np.min(np.asarray(allminstdlist[k])))
                    if all(item for item in rejflaglist):
                        continue
                    elif all(not item for item in rejflaglist):
                        print("CANNOT tell if the tube is big or small")
                        raise ValueError()
                    else:
                        tmppcd = tmppcdbelow90 if rejflaglist[0] else tmppcdover100
                        candidatetype = 2 if rejflaglist[0] else 1
                        tmpangles = np.arctan2(tmppcd[:, 1], tmppcd[:, 0])
                        tmpangles[tmpangles < 0] = 360 + tmpangles[tmpangles < 0]
                        if toggledebug:
                            print(np.std(tmpangles))
                            print("ACCEPTED! ID: ", i, j)
                        elearray[i][j] = candidatetype
                        eleconfidencearray[i][j] = 1
                        if toggledebug:
                            # normalized
                            objnp = p3dh.genpointcloudnodepath(tmppcd, pntsize=5)
                            rgb = np.random.rand(3)
                            objnp.setColor(rgb[0], rgb[1], rgb[2], 1)
                            objnp.reparentTo(base.render)
                            stick = p3dh.gendumbbell(spos=np.array([holepos[0], holepos[1], 10]),
                                                     epos=np.array([holepos[0], holepos[1], 60]))
                            stick.setColor(rgb[0], rgb[1], rgb[2], 1)
                            stick.reparentTo(base.render)
                            # original
                            tmppcd_tr = rm.homotransformpointarray(tubestand_homomat, tmppcd)
                            objnp_tr = p3dh.genpointcloudnodepath(tmppcd_tr, pntsize=5)
                            objnp_tr.setColor(rgb[0], rgb[1], rgb[2], 1)
                            objnp_tr.reparentTo(base.render)
                            spos_tr = rm.homotransformpoint(tubestand_homomat, np.array([holepos[0], holepos[1], 0]))
                            stick_tr = p3dh.gendumbbell(spos=np.array([spos_tr[0], spos_tr[1], 10]),
                                                        epos=np.array([spos_tr[0], spos_tr[1], 60]))
                            stick_tr.setColor(rgb[0], rgb[1], rgb[2], 1)
                            stick_tr.reparentTo(base.render)
                            # box normalized
                            center, bounds = rm.get_aabb(tmppcd)
                            boxextent = np.array(
                                [bounds[0, 1] - bounds[0, 0], bounds[1, 1] - bounds[1, 0], bounds[2, 1] - bounds[2, 0]])
                            boxhomomat = np.eye(4)
                            boxhomomat[:3, 3] = center
                            box = p3dh.genbox(extent=boxextent, homomat=boxhomomat,
                                              rgba=np.array([rgb[0], rgb[1], rgb[2], .3]))
                            box.reparentTo(base.render)
                            # box original
                            center_r = rm.homotransformpoint(tubestand_homomat, center)
                            boxhomomat_tr = copy.deepcopy(tubestand_homomat)
                            boxhomomat_tr[:3, 3] = center_r
                            box_tr = p3dh.genbox(extent=boxextent, homomat=boxhomomat_tr,
                                                 rgba=np.array([rgb[0], rgb[1], rgb[2], .3]))
                            box_tr.reparentTo(base.render)
                    if toggledebug:
                        print("------the new test is done------")
        return elearray, eleconfidencearray

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

            substracteddepth = self.bgdepth - fgdepth
            substracteddepth = substracteddepth.clip(50, 300)
            substracteddepth[substracteddepth == 50] = 0
            substracteddepth[substracteddepth == 300] = 0

            tempdepth = substracteddepth.flatten()
            objpcd = fgpcd[np.nonzero(tempdepth)]
            objpcd = self.transformcorrectedpcd(objpcd)
            if objpcdmerged is None:
                objpcdmerged = objpcd
            else:
                objpcdmerged = np.vstack((objpcdmerged, objpcd))

        # further crop x
        objpcdmerged = objpcdmerged[objpcdmerged[:,0]>200]

        return objpcdmerged

    def transformcorrectedpcd(self, pcdarray):
        """
        convert a poind cloud from its sensor frame to global frame

        :param pcdarray:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        return rm.homotransformpointarray(self.sensorhomomat, pcdarray)

    def gentubestand(self, homomat, rgba = np.array([0, .5, .7, 1.])):
        """

        :param homomat:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        tubestandcm = copy.deepcopy(self.tubestandcm)
        tubestandcm.set_homomat(homomat)
        tubestandcm.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

        return tubestandcm

    def gentubeandstandboxcm(self, homomat, wrapheight = 120, rgba = np.array([.5, .5, .5, .3])):
        """
        gen a solid box to wrap both a stand and the tubes in it

        :param homomat:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        homomat_copy = copy.deepcopy(homomat)
        homomat_copy[:3, 3] = homomat_copy[:3, 3] + homomat_copy[:3,2]* wrapheight/2
        tubeandstandboxcm = cm.CollisionModel(p3dh.genbox(np.array([self.tubestandsize[0], self.tubestandsize[1], 120]), homomat_copy))
        tubeandstandboxcm.setColor(rgba[0], rgba[1], rgba[2], rgba[3])

        return tubeandstandboxcm

    def gentubes(self, elearray, tubestand_homomat, eleconfidencearray=None, alpha=.3):
        """

        :param elearray:
        :param tubestand_homomat:
        :param eleconfidencearray: None by default
        :param alpha: only works when eleconfidencearray is None, it renders the array transparently
        :return:

        author: weiwei
        date: 20191229osaka
        """

        if eleconfidencearray is None:
            eleconfidencearray = np.ones_like(elearray) * alpha

        tubecmlist = []
        for i in range(5):
            for j in range(10):
                if elearray[i, j] == 1:
                    tubecm = self.tubebigcm
                    rgba = np.array([.7, .7, 0, eleconfidencearray[i, j]])
                elif elearray[i, j] == 2:
                    tubecm = self.tubesmallcm
                    rgba = np.array([.7, 0, .7, eleconfidencearray[i, j]])
                else:
                    continue
                newtubecm = copy.deepcopy(tubecm)
                tubemat = copy.deepcopy(tubestand_homomat)
                tubepos_normalized = np.array([self.tubeholecenters[i, j][0], self.tubeholecenters[i, j][1], 5])
                tubepos = rm.homotransformpoint(tubemat, tubepos_normalized)
                tubemat[:3, 3] = tubepos
                newtubecm.set_homomat(tubemat)
                newtubecm.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
                tubecmlist.append(newtubecm)

        return tubecmlist


if __name__ == '__main__':
    import robothelper
    import numpy as np
    import environment.collisionmodel as cm

    yhx = robothelper.RobotHelperX(usereal=False, startworld=True)
    loc = LocatorFixed(homomatfilename_start="rightfixture_light_homomat1", homomatfilename_goal="rightfixture_light_homomat2")

    # bgdepth = pickle.load(open("./databackground/bgdepth.pkl", "rb"))
    # bgpcd = pickle.load(open("./databackground/bgpcd.pkl", "rb"))

    # objpcdmerged = None
    # for i in range(1):
    #     yhx.pxc.triggerframe()
    #     fgdepth = yhx.pxc.getdepthimg()
    #     fgpcd = yhx.pxc.getpcd()
    #
    #     substracteddepth = bgdepth - fgdepth
    #     substracteddepth = substracteddepth.clip(40, 300)
    #     substracteddepth[substracteddepth == 40] = 0
    #     substracteddepth[substracteddepth == 300] = 0
    #
    #     # cv2.imshow("", yhx.pxc.cvtdepth(substracteddepth))
    #     # cv2.waitKey(0)
    #
    #     tempdepth = substracteddepth.flatten()
    #     objpcd = fgpcd[np.nonzero(tempdepth)]
    #     objpcd = loc.getcorrectedpcd(objpcd)
    #     if objpcdmerged is None:
    #         objpcdmerged = objpcd
    #     else:
    #         objpcdmerged = np.vstack((objpcdmerged, objpcd))
    objpcd = loc.capturecorrectedpcd(yhx.pxc, ncapturetimes=1)
    pcdnp = p3dh.genpointcloudnodepath(objpcd, pntsize=5)
    pcdnp.reparentTo(yhx.base.render)

    tbscm_start = loc.gentubestand(homomat=loc.tubestandhomomat_start)
    tbscm_start.reparentTo(yhx.base.render)
    tbscm_end = loc.gentubestand(homomat=loc.tubestandhomomat_goal)
    tbscm_end.reparentTo(yhx.base.render)

    elearray, eleconfidencearray = loc.findtubes(loc.tubestandhomomat_start, objpcd, toggledebug=False)
    # local axis
    yhx.p3dh.genframe(pos=loc.tubestandhomomat_start[:3,3], rotmat=loc.tubestandhomomat_start[:3,:3]).reparentTo(yhx.base.render)
    rbtnp = yhx.rbtmesh.genmnp(yhx.rbt)
    rbtnp.reparentTo(yhx.base.render)
    # pcdnp = p3dh.genpointcloudnodepath(objpcd, point_size=5)
    # pcdnp.reparentTo(yhx.base.render)
    # cornerhole_pcdnp = p3dh.genpointcloudnodepath(loc.calibrate_holes(objpcd), colors=np.array([1, 0, 0, 1]),
    #                                               point_size=10)
    # cornerhole_pcdnp.reparentTo(yhx.base.render)
    # positions, rotmats = loc.findtubestands_calibratewoodstickholes(objpcd)
    # for posrot in zip(positions, rotmats):
    #     loc.gentubestand(rm.homobuild(posrot[0], posrot[1])).reparentTo(yhx.base.render)

    # tbscm = loc.gentubestand(pos=pos)
    # tbscm.reparentTo(yhx.base.render)
    # tbscm.showcn()
    tubecms = loc.gentubes(elearray, loc.tubestandhomomat_start, eleconfidencearray=eleconfidencearray)
    for tbcm in tubecms:
        tbcm.reparentTo(yhx.base.render)
        tbcm.showcn()

    yhx.base.run()
