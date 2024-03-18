import pickle
import copy
import utiltools.thirdparty.o3dhelper as o3dh
import utiltools.robotmath as rm
import utiltools.misc.p3dhtils as p3dh


class Locator(object):

    def __init__(self, directory=None):
        if directory is None:
            self.bgdepth = pickle.load(open("../databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open("../databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open("../datacalibration/calibmat.pkl", "rb"))
            self.tstpcdnp = pickle.load(open("../dataobjtemplate/tubestandtemplatepcd.pkl", "rb"))# tstpcd, tube stand template
            self.tubestandcm = cm.CollisionModel("../objects/tubestand.stl")
            self.tubebigcm = cm.CollisionModel("../objects/tubebig_capped.stl", type="cylinder", ex_radius=0)
            self.tubesmallcm = cm.CollisionModel("../objects/tubesmall_capped.stl", type="cylinder", ex_radius=0)
        else:
            self.bgdepth = pickle.load(open(directory+"/databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open(directory+"/databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open(directory+"/datacalibration/calibmat.pkl", "rb"))
            self.tstpcdnp = pickle.load(open(directory+"/dataobjtemplate/tubestandtemplatepcd.pkl", "rb"))# tstpcd, tube stand template
            self.tubestandcm = cm.CollisionModel(directory+"/objects/tubestand.stl")
            self.tubebigcm = cm.CollisionModel(directory +"/objects/tubebig_capped.stl", type="cylinder", ex_radius=0)
            self.tubesmallcm = cm.CollisionModel(directory +"/objects/tubesmall_capped.stl", type="cylinder", ex_radius=0)

        self.tstpcdo3d = o3dh.nparray_to_o3dpcd(self.tstpcdnp)
        # down x, right y
        tubeholecenters = []
        for x in [-38,-19,0,19,38]:
            tubeholecenters.append([])
            for y in [-81, -63, -45, -27, -9, 9, 27, 45, 63, 81]:
                tubeholecenters[-1].append([x,y])
        self.tubeholecenters = np.array(tubeholecenters)
        self.tubeholesize = np.array([15, 16])

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
        tgtpcdo3d = o3dh.nparray_to_o3dpcd(tgtpcdnp)
        inlinnerrmse, homomat = o3dh.registration_icp_ptpt(self.tstpcdo3d, tgtpcdo3d, inithomomat, maxcorrdist=5, toggledebug=toggledebug)
        inithomomatflipped = copy.deepcopy(inithomomat)
        inithomomatflipped[:3,0] = -inithomomatflipped[:3,0]
        inithomomatflipped[:3,1] = -inithomomatflipped[:3,1]
        inlinnerrmseflipped, homomatflipped = o3dh.registration_icp_ptpt(self.tstpcdo3d, tgtpcdo3d, inithomomatflipped, maxcorrdist=5, toggledebug=toggledebug)
        # print(inlinnerrmse, inlinnerrmseflipped)
        if inlinnerrmseflipped < inlinnerrmse:
            homomat = homomatflipped
        return copy.deepcopy(homomat)

    def findtubestand_match(self, tgtpcdnp, toggledebug = False):
        """
        match self.tstpcd from tgtpcdnp
        NOTE: tgtpcdnp must be in global frame, use getglobalpcd to convert if local

        :param tgtpcdnp:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        tgtpcdo3d = o3dh.nparray_to_o3dpcd(tgtpcdnp)
        _, homomat = o3dh.registration_ptpln(self.tstpcdo3d, tgtpcdo3d, downsampling_voxelsize=5, toggledebug=toggledebug)

        return copy.deepcopy(homomat)

    def findtubestand_obb(self, tgtpcdnp, toggledebug = False):
        """
        match self.tstpcd from tgtpcdnp
        NOTE: tgtpcdnp must be in global frame, use getglobalpcd to convert if local

        :param tgtpcdnp:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        tgtpcdo3d = o3dh.nparray_to_o3dpcd(tgtpcdnp)
        tgtpcdo3d_removed = o3dh.remove_outlier(tgtpcdo3d, nb_points=50, radius=10)
        tgtpcdnp = o3dh.o3dpcd_to_parray(tgtpcdo3d_removed)

        # main axes
        tgtpcdnp2d = tgtpcdnp[:,:2] # TODO clip using sensor z
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
        homomat[:3,axind[0]] = np.array([vect[0,0], vect[1,0], 0])
        homomat[:3,axind[1]] = np.array([vect[0,1], vect[1,1], 0])
        homomat[:3,2] = np.array([0,0,1])
        if np.cross(homomat[:3,0], homomat[:3,1])[2] < -.5:
            homomat[:3,1] = -homomat[:3,1]
        homomat[:3, 3] = np.array([center[0], center[1], -15])
        return homomat

    def findtubes(self, tubestand_homomat, tgtpcdnp, toggledebug=False):
        """

        :param tubestand_homomat:
        :param tgtpcdnp:
        :return:
        """

        elearray = np.zeros((5,10))
        eleconfidencearray = np.zeros((5,10))

        tgtpcdo3d = o3dh.nparray_to_o3dpcd(tgtpcdnp)
        tgtpcdo3d_removed = o3dh.remove_outlier(tgtpcdo3d, downsampling_voxelsize=None, nb_points=90, radius=5)
        tgtpcdnp = o3dh.o3dpcd_to_parray(tgtpcdo3d_removed)
        # transform tgtpcdnp back
        tgtpcdnp_normalized = rm.homotransformpointarray(rm.homoinverse(tubestand_homomat), tgtpcdnp)
        if toggledebug:
            cm.CollisionModel(tgtpcdnp_normalized).reparentTo(base.render)
            tscm2 = cm.CollisionModel("../objects/tubestand.stl")
            tscm2.reparentTo(base.render)
        for i in range(5):
            for j in range(10):
                holepos = self.tubeholecenters[i][j]
                # squeeze the hole size by half
                tmppcd = tgtpcdnp_normalized[tgtpcdnp_normalized[:,0] < holepos[0]+self.tubeholesize[0]/1.9]
                tmppcd = tmppcd[tmppcd[:,0] > holepos[0]-self.tubeholesize[0]/1.9]
                tmppcd = tmppcd[tmppcd[:,1] < holepos[1]+self.tubeholesize[1]/1.9]
                tmppcd = tmppcd[tmppcd[:,1] > holepos[1]-self.tubeholesize[1]/1.9]
                tmppcd = tmppcd[tmppcd[:,2] > 70]
                if len(tmppcd) > 100:
                    print("------more than 100 raw points, start a new test------")
                    # use core tmppcd to decide tube types (avoid noises)
                    coretmppcd = tmppcd[tmppcd[:,0] < holepos[0]+self.tubeholesize[0]/4]
                    coretmppcd = coretmppcd[coretmppcd[:,0] > holepos[0]-self.tubeholesize[0]/4]
                    coretmppcd = coretmppcd[coretmppcd[:,1] < holepos[1]+self.tubeholesize[1]/4]
                    coretmppcd = coretmppcd[coretmppcd[:,1] > holepos[1]-self.tubeholesize[1]/4]
                    print("testing the number of core points...")
                    print(len(coretmppcd[:,2]))
                    if len(coretmppcd[:,2]) < 10:
                        print("------the new test is done------")
                        continue
                    if np.max(tmppcd[:,2]) > 100:
                        candidatetype = 1
                        tmppcd = tmppcd[tmppcd[:, 2] > 100] # crop tmppcd for better charge
                    else:
                        candidatetype = 2
                        tmppcd = tmppcd[tmppcd[:, 2] < 90]
                    if len(tmppcd) < 10:
                        continue
                    print("passed the core points test, rotate around...")
                    rejflag = False
                    for angle in np.linspace(0,180,20):
                        tmphomomat = np.eye(4)
                        tmphomomat[:3,:3] = rm.rodrigues(tubestand_homomat[:3, 2], angle)
                        newtmppcd = rm.homotransformpointarray(tmphomomat, tmppcd)
                        minstd = np.min(np.std(newtmppcd[:, :2], axis=0))
                        print(minstd)
                        if minstd<1.3:
                            rejflag = True
                    print("rotate round done")
                    if rejflag:
                        continue
                    else:
                        tmpangles = np.arctan2(tmppcd[:,1], tmppcd[:,0])
                        tmpangles[tmpangles<0]=360+tmpangles[tmpangles<0]
                        print(np.std(tmpangles))
                        print("ACCEPTED! ID: ", i, j)
                        elearray[i][j]= candidatetype
                        eleconfidencearray[i][j] = 1
                    if toggledebug:
                        # normalized
                        objnp = p3dh.genpointcloudnodepath(tmppcd, pntsize=5)
                        rgb = np.random.rand(3)
                        objnp.setColor(rgb[0], rgb[1], rgb[2], 1)
                        objnp.reparentTo(base.render)
                        stick = p3dh.gendumbbell(spos=np.array([holepos[0], holepos[1],10]), epos = np.array([holepos[0], holepos[1],60]))
                        stick.setColor(rgb[0], rgb[1], rgb[2], 1)
                        stick.reparentTo(base.render)
                        # original
                        tmppcd_tr = rm.homotransformpointarray(tubestand_homomat, tmppcd)
                        objnp_tr = p3dh.genpointcloudnodepath(tmppcd_tr, pntsize=5)
                        objnp_tr.setColor(rgb[0], rgb[1], rgb[2], 1)
                        objnp_tr.reparentTo(base.render)
                        spos_tr = rm.homotransformpoint(tubestand_homomat, np.array([holepos[0], holepos[1], 0]))
                        stick_tr = p3dh.gendumbbell(spos=np.array([spos_tr[0], spos_tr[1],10]), epos = np.array([spos_tr[0], spos_tr[1],60]))
                        stick_tr.setColor(rgb[0], rgb[1], rgb[2], 1)
                        stick_tr.reparentTo(base.render)
                        # box normalized
                        center, bounds = rm.get_aabb(tmppcd)
                        boxextent = np.array([bounds[0,1]-bounds[0,0], bounds[1,1]-bounds[1,0], bounds[2,1]-bounds[2,0]])
                        boxhomomat = np.eye(4)
                        boxhomomat[:3,3] = center
                        box = p3dh.genbox(extent=boxextent, homomat=boxhomomat, rgba=np.array([rgb[0], rgb[1], rgb[2], .3]))
                        box.reparentTo(base.render)
                        # box original
                        center_r = rm.homotransformpoint(tubestand_homomat, center)
                        boxhomomat_tr = copy.deepcopy(tubestand_homomat)
                        boxhomomat_tr[:3,3] = center_r
                        box_tr = p3dh.genbox(extent=boxextent, homomat=boxhomomat_tr, rgba=np.array([rgb[0], rgb[1], rgb[2], .3]))
                        box_tr.reparentTo(base.render)
                    print("------the new test is done------")
        return elearray, eleconfidencearray

    def capturecorrectedpcd(self, pxc, ncapturetimes = 1):
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
            substracteddepth = substracteddepth.clip(40, 300)
            substracteddepth[substracteddepth == 40] = 0
            substracteddepth[substracteddepth == 300] = 0

            tempdepth = substracteddepth.flatten()
            objpcd = fgpcd[np.nonzero(tempdepth)]
            objpcd = self.getcorrectedpcd(objpcd)
            if objpcdmerged is None:
                objpcdmerged = objpcd
            else:
                objpcdmerged = np.vstack((objpcdmerged, objpcd))

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
        tubestandcm.setColor(0,.5,.7,1.9)

        return tubestandcm

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
            eleconfidencearray = np.ones_like(elearray)*alpha

        tubecmlist = []
        for i in range(5):
            for j in range(10):
                if elearray[i,j] == 1:
                    tubecm = self.tubebigcm
                    rgba = np.array([.7, .7, 0, eleconfidencearray[i,j]])
                elif elearray[i,j] == 2:
                    tubecm = self.tubesmallcm
                    rgba = np.array([.7, 0, .7, eleconfidencearray[i,j]])
                else:
                    continue
                newtubecm = copy.deepcopy(tubecm)
                tubemat = copy.deepcopy(tubestand_homomat)
                tubepos_normalized = np.array([self.tubeholecenters[i,j][0], self.tubeholecenters[i,j][1], 5])
                tubepos  = rm.homotransformpoint(tubestand_homomat, tubepos_normalized)
                tubemat[:3, 3] = tubepos
                newtubecm.set_homomat(tubemat)
                newtubecm.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
                tubecmlist.append(newtubecm)

        return tubecmlist


if __name__ == '__main__':
    import robothelper
    import numpy as np
    import environment.collisionmodel as cm

    yhx = robothelper.RobotHelperX(usereal=False, startworld=False)
    yhx.startworld()
    loc = Locator()

    onscreennodepaths = [None]*100
    def estimate(yhx, lctr, onscreennodepaths, task):
        if yhx.base.inputmgr.keyMap['space'] is True:
            yhx.base.inputmgr.keyMap['space'] = False
            for idosnp, onscreennp in enumerate(onscreennodepaths):
                if onscreennp is not None:
                    onscreennp.removeNode()
                    onscreennodepaths[idosnp] = None
                else:
                    break
            objpcd = lctr.capturecorrectedpcd(yhx.pxc)
            homomat = loc.findtubestand_matchonobb(objpcd, toggledebug=False)

            elearray, eleconfidencearray = loc.findtubes(homomat, objpcd, toggledebug=False)
            framenp = yhx.p3dh.genframe(pos=homomat[:3,3], rotmat=homomat[:3,:3])
            framenp.reparentTo(yhx.base.render)
            onscreennodepaths[0] = framenp
            rbtnp = yhx.rbtmesh.genmnp(yhx.robot_s)
            rbtnp.reparentTo(yhx.base.render)
            onscreennodepaths[1] = rbtnp
            pcdnp = p3dh.genpointcloudnodepath(objpcd, pntsize=5)
            pcdnp.reparentTo(yhx.base.render)
            onscreennodepaths[2] = pcdnp

            tbscm = loc.gentubestand(homomat=homomat)
            tbscm.reparentTo(yhx.base.render)
            tbscm.showcn()
            onscreennodepaths[3] = tbscm
            tubecms = loc.gentubes(elearray, tubestand_homomat=homomat, eleconfidencearray=eleconfidencearray)
            for i, tbcm in enumerate(tubecms):
                tbcm.reparentTo(yhx.base.render)
                hmat = tbcm.get_homomat()
                hmat[:3, 3] -= hmat[:3,2]*50
                tbcm.set_homomat(hmat)
                tbcm.showcn()
                onscreennodepaths[i+4] = tbcm
            return task.again
        else:
            return task.again

    taskMgr.doMethodLater(0.04, estimate, "estimate",
                          extraArgs=[yhx, loc, onscreennodepaths],
                          appendTask=True)
    yhx.rbtmesh.genmnp(yhx.robot_s).reparentTo(base.render)
    yhx.base.run()
