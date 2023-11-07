import pickle
import copy
import utiltools.thirdparty.o3dhelper as o3dh
import utiltools.robotmath as rm
import utiltools.thirdparty.p3dhelper as p3dh
import numpy as np
import environment.collisionmodel as cm

class TLocator(object):

    def __init__(self, directory=None):
        """

        :param directory:
        """

        srcpcdfilename = "vacuumhead_templatepcd.pkl"
        srcmeshfilename = "vacuumhead.stl"

        if directory is None:
            self.bgdepth = pickle.load(open("./databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open("./databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open("./datacalibration/calibmat.pkl", "rb"))
            self.srcpcdnp = pickle.load(open("./dataobjtemplate/"+srcpcdfilename, "rb"))# tstpcd, tube stand template
            self.srccm = cm.CollisionModel("./objects/"+srcmeshfilename)
        else:
            self.bgdepth = pickle.load(open(directory+"/databackground/bgdepth.pkl", "rb"))
            self.bgpcd = pickle.load(open(directory+"/databackground/bgpcd.pkl", "rb"))
            self.sensorhomomat = pickle.load(open(directory+"/datacalibration/calibmat.pkl", "rb"))
            self.srcpcdnp = pickle.load(open(directory+"/dataobjtemplate/"+srcpcdfilename, "rb"))# tstpcd, tube stand template
            self.srccm = cm.CollisionModel(directory+"/objects/"+srcmeshfilename)

        # for compatibility with locatorfixed
        self.objhomomat =  None

    def findobj(self, tgtpcdnp, toggledebug=False):
        """
        match self.tstpcd from tgtpcdnp
        an obb-based initialization is performed before icp

        :param tgtpcdnp:
        :param toggledebug:
        :return:

        author: weiwei
        date:20191229osaka
        """

        # toggle the following command to crop the point cloud
        # tgtpcdnp = tgtpcdnp[np.logical_and(tgtpcdnp[:,2]>40, tgtpcdnp[:,2]<60)]

        # 20200425 cluster is further included
        pcdarraylist, _ = o3dh.cluster_pcd(tgtpcdnp)
        tgtpcdnp = max(pcdarraylist, key = lambda x:len(x))
        # for pcdarray in pcdarraylist:
        #     rgb = np.random.rand(3)
        #     rgba = np.array([rgb[0], rgb[1], rgb[2], 1])
        #     pcdnp = p3dh.genpointcloudnodepath(pcdarray, point_size=5, colors=rgba)
        #     pcdnp.reparentTo(base.render)
        #     break
        # base.run()

        inlinnerrmse, homomat = o3dh.registration_ptpt(self.srcpcdnp, tgtpcdnp, toggledebug=toggledebug)
        self.tubestandhomomat =  homomat

        return copy.deepcopy(homomat)

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
            substracteddepth = substracteddepth.clip(15, 300)
            substracteddepth[substracteddepth == 15] = 0
            substracteddepth[substracteddepth == 300] = 0

            tempdepth = substracteddepth.flatten()
            objpcd = fgpcd[np.nonzero(tempdepth)]
            objpcd = self.getcorrectedpcd(objpcd)
            if objpcdmerged is None:
                objpcdmerged = objpcd
            else:
                objpcdmerged = np.vstack((objpcdmerged, objpcd))

        # further crop x
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

    def genobjcm(self, homomat):
        """

        :param homomat:
        :return:

        author: weiwei
        date: 20200522osaka
        """

        objcm = copy.deepcopy(self.srccm)
        objcm.set_homomat(homomat)
        objcm.setColor(.5,.5,.5,1)

        return objcm


if __name__ == '__main__':
    import robothelper
    import numpy as np
    import environment.collisionmodel as cm

    yhx = robothelper.RobotHelperX(usereal=False, startworld=True)
    loc = TLocator(directory='..')

    objpcd = loc.capturecorrectedpcd(yhx.pxc, ncapturetimes=1)
    homomat = loc.findobj(objpcd, toggledebug=False)
    loc.genobjcm(homomat).reparentTo(yhx.base.render)
    pcdnp = p3dh.genpointcloudnodepath(objpcd, pntsize=2)
    pcdnp.reparentTo(yhx.base.render)
    yhx.base.run()


    # pos = loc.findtubestand_match(objpcdmerged, toggle_dbg=True)

    elearray, eleconfidencearray = loc.findtubes(homomat, objpcd, toggledebug=False)
    yhx.p3dh.genframe(pos=homomat[:3,3], rotmat=homomat[:3,:3]).reparentTo(yhx.base.render)
    rbtnp = yhx.rbtmesh.genmnp(yhx.robot_s)
    rbtnp.reparentTo(yhx.base.render)
    pcdnp = p3dh.genpointcloudnodepath(objpcd, pntsize=5)
    pcdnp.reparentTo(yhx.base.render)

    tbscm = loc.gentubestand(homomat=homomat)
    tbscm.reparentTo(yhx.base.render)
    tbscm.showcn()
    tubecms = loc.gentubes(elearray, tubestand_homomat=homomat, eleconfidencearray=eleconfidencearray)
    for tbcm in tubecms:
        tbcm.reparentTo(yhx.base.render)
        tbcm.showcn()

    yhx.base.run()
