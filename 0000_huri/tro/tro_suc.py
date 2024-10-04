import pickle
import utiltools.thirdparty.o3dhelper as o3dh
import utiltools.robotmath as rm
import numpy as np


class PcdGrab(object):

    def __init__(self, directory=None):
        if directory is None:
            self.bgdepth = pickle.load(open("../databackground/bgdepthlow.pkl", "rb"))
            self.bgpcd = pickle.load(open("../databackground/bgpcddepthlow.pkl", "rb"))
            self.sensorhomomat = pickle.load(open("../datacalibration/calibmat.pkl", "rb"))
        else:
            self.bgdepth = pickle.load(open(directory+"/databackground/bgdepthlow.pkl", "rb"))
            self.bgpcd = pickle.load(open(directory+"/databackground/bgpcddepthlow.pkl", "rb"))
            self.sensorhomomat = pickle.load(open(directory+"/datacalibration/calibmat.pkl", "rb"))

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
            substracteddepth = substracteddepth.clip(20, 300)
            substracteddepth[substracteddepth == 20] = 0
            substracteddepth[substracteddepth == 300] = 0
            substracteddepth[:100, :] = 0 # 300, 1700 for high resolution
            substracteddepth[1000:, :] = 0
            substracteddepth[:, :100] = 0
            substracteddepth[:, 1000:] = 0

            tempdepth = substracteddepth.flatten()
            objpcd = fgpcd[np.nonzero(tempdepth)]
            objpcd = self.getcorrectedpcd(objpcd)
            if objpcdmerged is None:
                objpcdmerged = objpcd
            else:
                objpcdmerged = np.vstack((objpcdmerged, objpcd))

            tgtpcdo3d = o3dh.nparray_to_o3dpcd(objpcdmerged)
            tgtpcdo3d_removed = o3dh.remove_outlier(tgtpcdo3d, nb_points=50, radius=10)
            tgtpcdnp = o3dh.o3dpcd_to_parray(tgtpcdo3d_removed)

        return tgtpcdnp

    def getcorrectedpcd(self, pcdarray):
        """
        convert a poind cloud from its sensor frame to global frame

        :param pcdarray:
        :return:

        author: weiwei
        date: 20191229osaka
        """

        return rm.homotransformpointarray(self.sensorhomomat, pcdarray)

if __name__ == '__main__':
    import robothelper
    import utiltools.misc.p3dhtils as p3dh
    import environment.collisionmodel as cm
    from wrs import manipulation as fs, manipulation as hlb

    rhx = robothelper.RobotHelperX(usereal=True, startworld=True)
    # rhx.movetox(rhx.robot_s.initrgtjnts, arm_name="rgt")
    # rhx.movetox(rhx.robot_s.initlftjnts, arm_name="lft")
    # rhx.closegripperx(arm_name="rgt")
    # rhx.closegripperx(arm_name="lft")
    # rhx.opengripperx(arm_name="rgt")
    # rhx.opengripperx(arm_name="lft")

    pg = PcdGrab()
    nppcd = pg.capturecorrectedpcd(pxc=rhx.pxc)

    p3dpcd = p3dh.genpointcloudnodepath(nppcd, pntsize=1.57)
    # p3dpcd.reparentTo(rhx.base.render)

    effa = hlb.EFFactory()
    freesuctst = fs.Freesuc()
    reconstructedtrimeshlist, nppcdlist = o3dh.reconstruct_surfaces_bp(nppcd, radii=(10, 12))
    # for i, tmpnppcd in enumerate(nppcdlist):
    #     p3dpcd = p3dh.genpointcloudnodepath(tmpnppcd, point_size=1.57)
    #     p3dpcd.reparentTo(rhx.base.render)
    #     if i == 0:
    #         p3dpcd.setColor(.7,0,0,1)
    #     elif i == 1:
    #         p3dpcd.setColor(0,0,.7,1)
    #     elif i == 2:
    #         p3dpcd.setColor(0,.7,0,1)
    #     else:
    #         p3dpcd.setColor(1,1,1,1)
    for i, reconstructedtrimesh in enumerate(reconstructedtrimeshlist):
        reconstructedmeshobjcm = cm.CollisionModel(reconstructedtrimesh)
        reconstructedmeshobjcm.reparentTo(rhx.base.render)
        if i == 0:
            reconstructedmeshobjcm.setColor(.7,0,0,1)
        elif i == 1:
            reconstructedmeshobjcm.setColor(0,0,.7,1)
        elif i == 2:
            reconstructedmeshobjcm.setColor(0,.7,0,1)
        else:
            reconstructedmeshobjcm.setColor(1,1,1,1)

        freesuctst.plansuctions(effa=effa, objinit=reconstructedmeshobjcm, faceangle=.85, segangle=.85, mindist=10, reduceradius=30, discretesize=8, torqueresist = 100)
        # freesuctst.showfacets(togglesamples=True, togglenormals=False,
        #                       togglesamples_ref=True, togglenormals_ref=False,
        #                       togglesamples_refcls=True, togglenormals_refcls=False, specificfacet=True)
        p3dh.gensphere(pos=np.mean(freesuctst._cmodel.trm_mesh.vertices, axis=0), radius=5, rgba=[1, 1, 1, 1]).reparentTo(rhx.base.render)
        print(len(freesuctst.sucrotmats_planned))
        print(len(freesuctst.facets))
        for i, homomat in enumerate(freesuctst.sucrotmats_planned):
            # pos[:3,3] = pos[:3,3]-pos[:3,2]*120
            homomatnew = np.copy(homomat)
            homomatnew[:3,3] = homomat[:3,3]-homomat[:3,2]*3
            homomatnew[:3, :3] = np.dot(rm.rodrigues(homomat[:3,0], 45), homomat[:3,:3])
            # tmpef = effa.genendeffector()
            # tmpef.sethomomat(pos)
            # tmpef.reparentTo(rhx.base.render)
            # tmpef.setcolor(1, 1, 1, .3)
            armjnts = rhx.movetopose(homomatnew, "lft")
            if armjnts is not None:
                # if i == 1:
                tmpef = effa.genendeffector()
                tmpef.set_homomat(homomat)
                tmpef.reparentTo(rhx.base.render)
                tmpef.set_rgba(1, 1, 1, .3)
                pos = homomat[:3,3]
                rot = homomat[:3,:3]
                rhx.opengripperx("lft")
                rhx.movetox(armjnts, "lft")
                break
                # break
        # for i, pos in enumerate(freesuctst.sucrotmats_removed):
        #     # if i == 1:
        #     tmpef = effa.genendeffector()
        #     tmpef.sethomomat(pos)
        #     tmpef.reparentTo(rhx.base.render)
        #     tmpef.setcolor(1, 0, 0, .3)
        break
    rhx.show()


