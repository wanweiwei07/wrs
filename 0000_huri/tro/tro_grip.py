import pickle
import utiltools.thirdparty.o3dhelper as o3dh
import utiltools.robotmath as rm
import numpy as np


class PcdGrab(object):

    def __init__(self, directory=None):
        if directory is None:
            self.bgdepth1 = pickle.load(open("../databackground/tro_bgdepthlow1.pkl", "rb"))
            self.bgpcd1 = pickle.load(open("../databackground/tro_bgpcddepthlow1.pkl", "rb"))
            self.bgdepth2 = pickle.load(open("../databackground/tro_bgdepthlow2.pkl", "rb"))
            self.bgpcd2 = pickle.load(open("../databackground/tro_bgpcddepthlow2.pkl", "rb"))
            self.bgdepth3 = pickle.load(open("../databackground/tro_bgdepthlow3.pkl", "rb"))
            self.bgpcd3 = pickle.load(open("../databackground/tro_bgpcddepthlow3.pkl", "rb"))
            self.bgdepth4 = pickle.load(open("../databackground/tro_bgdepthlow4.pkl", "rb"))
            self.bgpcd4 = pickle.load(open("../databackground/tro_bgpcddepthlow4.pkl", "rb"))
            self.sensorhomomat = pickle.load(open("../datacalibration/calibmat.pkl", "rb"))
        else:
            self.bgdepth1 = pickle.load(open(directory+"/databackground/tro_bgdepthlow1.pkl", "rb"))
            self.bgpcd1 = pickle.load(open(directory+"/databackground/tro_bgpcddepthlow1.pkl", "rb"))
            self.bgdepth2 = pickle.load(open(directory+"/databackground/tro_bgdepthlow2.pkl", "rb"))
            self.bgpcd2 = pickle.load(open(directory+"/databackground/tro_bgpcddepthlow2.pkl", "rb"))
            self.bgdepth3 = pickle.load(open(directory+"/databackground/tro_bgdepthlow3.pkl", "rb"))
            self.bgpcd3 = pickle.load(open(directory+"/databackground/tro_bgpcddepthlow3.pkl", "rb"))
            self.bgdepth4 = pickle.load(open(directory+"/databackground/tro_bgdepthlow4.pkl", "rb"))
            self.bgpcd4 = pickle.load(open(directory+"/databackground/tro_bgpcddepthlow4.pkl", "rb"))
            self.sensorhomomat = pickle.load(open(directory+"/datacalibration/calibmat.pkl", "rb"))

    def capturecorrectedpcd(self, pxc, ncapturetimes=1, id = 1):
        """
        capture a poind cloud and transform it from its sensor frame to global frame

        :param pcdnp:
        :return:

        author: weiwei
        date: 20200108
        """

        bgdepth = self.bgdepth1
        if id == 2:
            bgdepth = self.bgdepth2
        elif id == 3:
            bgdepth = self.bgdepth3
        elif id == 4:
            bgdepth = self.bgdepth4


        objpcdmerged = None
        for i in range(ncapturetimes):
            pxc.triggerframe()
            fgdepth = pxc.getdepthimg()
            fgpcd = pxc.getpcd()

            substracteddepth = bgdepth - fgdepth
            substracteddepth = substracteddepth.clip(50, 600)
            substracteddepth[substracteddepth == 50] = 0
            substracteddepth[substracteddepth == 600] = 0
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
    from wrs import manipulation as fg, manipulation as yi
    import trimesh

    pg = PcdGrab()
    rhx = robothelper.RobotHelperX(usereal=True, startworld=True)
    eepos = np.array([400, 0, 200])
    eerot=np.array([[1,0,0],[0,0,-1],[0,1,0]]).T
    armjnts= rhx.movetoposrot(eepos = eepos, eerot=eerot, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    nppcd1 = pg.capturecorrectedpcd(pxc=rhx.pxc, id=1)-eepos
    eerot2 = np.dot(rm.rodrigues([0,1,0], 90), eerot)
    armjnts= rhx.movetoposrot(eepos=eepos, eerot=eerot2, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    nppcd2 = pg.capturecorrectedpcd(pxc=rhx.pxc, id=2)-eepos
    eerot3 = np.dot(rm.rodrigues([0,1,0], 180), eerot)
    armjnts= rhx.movetoposrot(eepos=eepos, eerot=eerot3, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    nppcd3 = pg.capturecorrectedpcd(pxc=rhx.pxc, id=3)-eepos
    eerot4 = np.dot(rm.rodrigues([0,1,0], 270), eerot)
    armjnts= rhx.movetoposrot(eepos=eepos, eerot=eerot4, armname="rgt")
    rhx.movetox(armjnts, "rgt")
    nppcd4 = pg.capturecorrectedpcd(pxc=rhx.pxc, id=4)-eepos

    mergednppcd = nppcd4
    mergedlistnrmls = [[0,0,1]]*len(mergednppcd)
    mergednppcd = o3dh.merge_pcd(mergednppcd, nppcd1, rm.rodrigues([0, 1, 0], 270), posmat2=np.zeros(3))
    mergedlistnrmls += [[-1,0,0]]*len(nppcd1)
    mergednppcd = o3dh.merge_pcd(mergednppcd, nppcd2, rm.rodrigues([0, 1, 0], 180), posmat2=np.zeros(3))
    mergedlistnrmls += [[0,0,-1]]*len(nppcd2)
    mergednppcd = o3dh.merge_pcd(mergednppcd, nppcd3, rm.rodrigues([0, 1, 0], 90), posmat2=np.zeros(3))
    mergedlistnrmls += [[1,0,0]]*len(nppcd3)
    mergednppcdnrmls = np.array(mergedlistnrmls)
    mergednppcd += eepos
    p3dpcd = p3dh.genpointcloudnodepath(mergednppcd, pntsize=1.57)
    # p3dpcd.reparentTo(rhx.base.render)
    # rhx.show()

    yifa = yi.YumiIntegratedFactory()
    reconstructedtrimeshlist, nppcdlist = o3dh.reconstruct_surfaces_bp(mergednppcd, mergednppcdnrmls, radii=(7, 10))
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
        # reconstructedmeshobjcm.reparentTo(rhx.base.render)
        # if i == 1:
        #     reconstructedmeshobjcm.setColor(.7,0,0,1)
        # elif i == 0:
        #     reconstructedmeshobjcm.setColor(0,0,.7,1)
        # elif i == 2:
        #     reconstructedmeshobjcm.setColor(0,.7,0,1)
        # else:
        #     reconstructedmeshobjcm.setColor(1,1,1,1)

        freegriptst = fg.Freegrip(reconstructedmeshobjcm, yifa.genHand(), faceangle=.9, segangle=.9, refine1min=7, togglebcdcdebug=True, useoverlap=True)
        # geom = None
        facetsizes = []
        for i, faces in enumerate(freegriptst.facets):
            rgba = [np.random.random(), np.random.random(), np.random.random(), 1]
            tm = trimesh.Trimesh(vertices=freegriptst.trm_mesh.vertices, faces=freegriptst.trm_mesh.faces[faces],
                                 face_normals=freegriptst.trm_mesh.face_normals[faces])
            facetcm = cm.CollisionModel(objinit=tm)
            facetcm.setColor(rgba[0], rgba[1], rgba[2], rgba[3])
            facetcm.reparentTo(rhx.base.render)
            facetsizes.append(tm.area)

        freegriptst.plangrasps(contactoffset=2)

        ngrasp = 0
        for pfacets in freegriptst.griprotmats_planned:
            for gmats in pfacets:
                ngrasp += len(gmats)
        nsamples = 0
        for i in range(len(freegriptst.facets)):
            for samples in freegriptst.objsamplepnts_refcls[i]:
                nsamples += len(samples)
        ncpairs = 0
        for pfacets in freegriptst.gripcontactpairs:
            for cpairs in pfacets:
                ncpairs += len(cpairs)
        print("number of grasps planned", ngrasp)
        print("number of samples", nsamples)
        print("number of contactpairs", ncpairs)

        for i, freegriprotmat in enumerate(freegriptst.griprotmats_planned):
            armjnts = rhx.movetopose(freegriprotmat, "lft")
            if armjnts is not None:
                tmpef = yifa.genHand()
                tmpef.set_homomat(freegriprotmat)
                tmpef.reparentTo(rhx.base.render)
                tmpef.setColor(1, 1, 1, .3)
                pos = freegriprotmat[:3, 3]
                rot = freegriprotmat[:3, :3]
                rhx.opengripperx("lft")
                rhx.movetox(armjnts, "lft")
                break
            # print(freegriprotmat)
            # hand = yifa.genHand()
            # hand.setColor(1, 1, 1, .3)
            # newpos = freegriprotmat[:3,3] - freegriprotmat[:3,2] * 0.0
            # freegriprotmat[:3,3] = newpos
            # hand.sethomomat(freegriprotmat)
            # hand.setjawwidth(freegriptst.gripjawwidth_planned[i])
            # hand.reparentTo(rhx.base.render)
    rhx.show()


