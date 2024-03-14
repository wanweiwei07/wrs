import itertools
import pickle
import os
import environment.bulletcdhelper as bch

class HandoverPlanner(object):
    """

    author: hao chen, ruishuang liu, refactored by weiwei
    date: 20191122
    """

    def __init__(self, obj, rhx, retractdistance=100):
        """

        :param obj: obj name (str) or obj_cmodel, obj_cmodel is for debug purpose
        :param rhx: see helper.py
        :param retractdistance: retraction linear_distance

        author: hao, refactored by weiwei
        date: 20191206, 20200104osaka
        """

        if isinstance(obj, str):
            self.objname = obj
        elif isinstance(obj, rhx.mcm.CollisionModel):
            self.objname = obj.name
            self.objcm = obj
        self.rhx = rhx
        self.rbt = rhx.robot_s
        self.retractdistance = retractdistance
        self.bcdchecker = bch.MCMchecker(toggledebug=False)
        with open(os.path.join(rhx.path, "grasp" + rhx.rgthndfa.name, "predefinedgrasps.pickle"), "rb") as file:
            graspdata = pickle.load(file)
            self.identityglist_rgt = graspdata[self.objname]
        with open(os.path.join(rhx.path, "grasp" + rhx.lfthndfa.name, "predefinedgrasps.pickle"), "rb") as file:
            graspdata = pickle.load(file)
            self.identityglist_lft = graspdata[self.objname]
        self.rgthndfa = rhx.rgthndfa
        self.lfthndfa = rhx.lfthndfa

        # paramters
        self.fpsnpmat4 = []
        self.identitygplist = [] # grasp pair list at the identity pose
        self.fpsnestedglist_rgt = {} # fpsnestedglist_rgt[fpid] = [g0, g1, ...], fpsnestedglist means glist at each floating pose
        self.fpsnestedglist_lft = {} # fpsnestedglist_lft[fpid] = [g0, g1, ...]
        self.ikfid_fpsnestedglist_rgt = {} # fid - feasible id
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_lft = {}

    def genhvgpsgl(self, posvec, rotmat = None):
        """
        generate the handover grasps using the given position and orientation
        sgl means a single position
        rotmat could either be a single one or multiple (0,90,180,270, default)

        :param posvec
        :param rotmat
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identitygplist = []
        if rotmat is None:
            self.fpsnpmat4 = self.rhx.rm.gen_icohomomats_flat(posvec=posvec, angles=[0, 90, 180, 270])
        else:
            self.fpsnpmat4 = [self.rhx.rm.homobuild(posvec, rotmat)]
        self.__genidentitygplist()
        self.__genfpsnestedglist()
        self.__checkik()

        if not os.path.exists(os.path.join(rhx.root, "datahandover")):
            os.mkdir(os.path.join(rhx.root, "datahandover"))
        with open(os.path.join(rhx.root, "datahandover", self.objname + "_hndovrinfo.pickle"), "wb") as file:
            pickle.dump([self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft,
                         self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft,
                         self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft], file)

    def genhvgplist(self, hvgplist):
        """
        generate the handover grasps using the given list of pos

        :param hvgplist, [homomat0, homomat1, ...]
        :return: data is saved as a file

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        self.identitygplist = []
        self.fpsnpmat4 = hvgplist
        self.__genidentitygplist()
        self.__genfpsnestedglist()
        self.__checkik()

        if not os.path.exists(os.path.join(rhx.root, "datahandover")):
            os.mkdir(os.path.join(rhx.root, "datahandover"))
        with open(os.path.join(rhx.root, "datahandover", self.objname + "_hndovrinfo.pickle"), "wb") as file:
            pickle.dump([self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft,
                         self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft,
                         self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft], file)

    def gethandover(self):
        """
        io interface to load the previously planned data

        :return:

        author: hao, refactored by weiwei
        date: 20191206, 20191212
        """

        with open(os.path.join(rhx.root, "datahandover", self.objname + "_hndovrinfo.pickle"), "rb") as file:
            self.fpsnpmat4, self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lftt, \
            self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
            self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft = pickle.load(file)

        return self.identityglist_rgt, self.identityglist_lft, self.fpsnpmat4, \
               self.identitygplist, self.fpsnestedglist_rgt, self.fpsnestedglist_lft, \
               self.ikfid_fpsnestedglist_rgt, self.ikfid_fpsnestedglist_lft, \
               self.ikjnts_fpsnestedglist_rgt, self.ikjnts_fpsnestedglist_lft

    def __genidentitygplist(self):
        """
        fill up self.identitygplist

        :return:

        author: weiwei
        date: 20191212
        """

        rgthnd = self.rgthndfa.genHand()
        lfthnd = self.lfthndfa.genHand()
        pairidlist = list(itertools.product(range(len(self.identityglist_rgt)), range(len(self.identityglist_lft))))
        for i in range(len(pairidlist)):
            print("generating identity gplist...", i, len(pairidlist))
            # Check whether the hands collide with each or not
            ir, il = pairidlist[i]
            rgthnd.setMat(base.pg.np4ToMat4(self.identityglist_rgt[ir][2]))
            rgthnd.setjawwidth(self.identityglist_rgt[ir][0])
            lfthnd.setMat(base.pg.np4ToMat4(self.identityglist_lft[il][2]))
            lfthnd.setjawwidth(self.identityglist_lft[il][0])
            ishndcollided = self.bcdchecker.isMeshListMeshListCollided(rgthnd.cmlist, lfthnd.cmlist)
            if not ishndcollided:
                self.identitygplist.append(pairidlist[i])

    def __genfpsnestedglist(self):
        """
        generate the grasp list for the floating poses

        :return:

        author: hao chen, revised by weiwei
        date: 20191122
        """

        self.fpsnestedglist_rgt = {}
        self.fpsnestedglist_lft = {}
        for posid, icomat4 in enumerate(self.fpsnpmat4):
            print("generating nested glist at the floating poses...", posid, len(self.fpsnpmat4))
            glist = []
            for jawwidth, fc, homomat in self.identityglist_rgt:
                tippos = self.rhx.rm.homotransformpoint(icomat4, fc)
                homomat = self.rhx.np.dot(icomat4, homomat)
                glist.append([jawwidth, tippos, homomat])
            self.fpsnestedglist_rgt[posid] = glist
            glist = []
            for jawwidth, fc, homomat in self.identityglist_lft:
                tippos = self.rhx.rm.homotransformpoint(icomat4, fc)
                homomat = self.rhx.np.dot(icomat4, homomat)
                glist.append([jawwidth, tippos, homomat])
            self.fpsnestedglist_lft[posid] = glist

    def __checkik(self):
        # Check the IK of both hand in the handover pose
        ### right hand
        self.ikfid_fpsnestedglist_rgt = {}
        self.ikjnts_fpsnestedglist_rgt = {}
        self.ikfid_fpsnestedglist_lft = {}
        self.ikjnts_fpsnestedglist_lft = {}
        for posid in self.fpsnestedglist_rgt.keys():
            armname = 'rgt'
            fpglist_thispose = self.fpsnestedglist_rgt[posid]
            for i, [_, tippos, homomat] in enumerate(fpglist_thispose):
                hndrotmat4 = homomat
                fgrcenternp = tippos
                fgrcenterrotmatnp = hndrotmat4[:3, :3]
                handa = -hndrotmat4[:3, 2]
                minusworldy = self.rhx.np.array([0,-1,0])
                if self.rhx.rm.degree_betweenvector(handa, minusworldy) < 90:
                    msc = self.rbt.numik(fgrcenternp, fgrcenterrotmatnp, armname)
                    if msc is not None:
                        fgrcenternp_handa = fgrcenternp + handa * self.retractdistance
                        msc_handa = self.rbt.numikmsc(fgrcenternp_handa, fgrcenterrotmatnp, msc, armname)
                        if msc_handa is not None:
                            if posid not in self.ikfid_fpsnestedglist_rgt:
                                self.ikfid_fpsnestedglist_rgt[posid] = []
                            self.ikfid_fpsnestedglist_rgt[posid].append(i)
                            if posid not in self.ikjnts_fpsnestedglist_rgt:
                                self.ikjnts_fpsnestedglist_rgt[posid] = {}
                            self.ikjnts_fpsnestedglist_rgt[posid][i] = [msc, msc_handa]
        ### left hand
        for posid in self.fpsnestedglist_lft.keys():
            armname = 'lft'
            fpglist_thispose = self.fpsnestedglist_lft[posid]
            for i, [_, tippos, homomat] in enumerate(fpglist_thispose):
                hndrotmat4 = homomat
                fgrcenternp = tippos
                fgrcenterrotmatnp = hndrotmat4[:3, :3]
                handa = -hndrotmat4[:3, 2]
                plusworldy = self.rhx.np.array([0,1,0])
                if self.rhx.rm.degree_betweenvector(handa, plusworldy) < 90:
                    msc = self.rbt.numik(fgrcenternp, fgrcenterrotmatnp, armname)
                    if msc is not None:
                        fgrcenternp_handa = fgrcenternp + handa * self.retractdistance
                        msc_handa = self.rbt.numikmsc(fgrcenternp_handa, fgrcenterrotmatnp, msc, armname)
                        if msc_handa is not None:
                            if posid not in self.ikfid_fpsnestedglist_lft:
                                self.ikfid_fpsnestedglist_lft[posid] = []
                            self.ikfid_fpsnestedglist_lft[posid].append(i)
                            if posid not in self.ikjnts_fpsnestedglist_lft:
                                self.ikjnts_fpsnestedglist_lft[posid] = {}
                            self.ikjnts_fpsnestedglist_lft[posid][i] = [msc, msc_handa]

    def checkhndenvcollision(self, homomat, obstaclecmlist, armname="rgt"):
        """

        :param homomat:
        :param obstaclecmlist:
        :return:

        author: ruishuang
        date: 20191122
        """

        if armname == "rgt":
            handtmp = self.rgthndfa.genHand()
        else:
            handtmp = self.lfthndfa.genHand()
        handtmp.set_homomat(homomat)
        handtmp.setjawwidth(handtmp.jawwidthopen)
        iscollided = self.bcdchecker.isMeshListMeshListCollided(handtmp.cmlist, obstaclecmlist)

        return iscollided

if __name__ == "__main__":
    import robothelper as yh

    rhx = yh.RobotHelperX(usereal=False)
    base = rhx.startworld()

    objcm = rhx.cm.CollisionModel("./objects/tubebig.stl")

    hmstr = HandoverPlanner(obj=objcm, rhx=rhx, retractdistance=100)
    hmstr.genhvgpsgl(rhx.np.array([400, 0, 300]), rhx.np.eye(3))

    identityglist_rgt, identityglist_lft, fpsnpmat4, identitygplist, fpsnestedglist_rgt, fpsnestedglist_lft, \
    ikfid_fpsnestedglist_rgt,ikfid_fpsnestedglist_lft, \
    ikjnts_fpsnestedglist_rgt, ikjnts_fpsnestedglist_lft = hmstr.gethandover()
    print(ikfid_fpsnestedglist_lft.keys())
    print(ikfid_fpsnestedglist_rgt.keys())

    poseid = 0
    for gp in identitygplist:
        ir = gp[0]
        il = gp[1]
        if poseid not in ikfid_fpsnestedglist_lft.keys() or poseid not in ikfid_fpsnestedglist_rgt:
            poseid += 1
            print("Poseid " + str(poseid) + " do not have feasible ik solution, continue...")
            continue
        if ir in ikfid_fpsnestedglist_rgt[poseid] and il in ikfid_fpsnestedglist_lft[poseid]:
            jawwidth_rgt, fc_rgt, rotmat_rgt = fpsnestedglist_rgt[poseid][ir]
            jnts_rgt, jnts_msc_rgt = ikjnts_fpsnestedglist_rgt[poseid][ir]
            jawwidth_lft, fc_lft, rotmat_lft = fpsnestedglist_rgt[poseid][il]
            jnts_lft, jnts_msc_lft = ikjnts_fpsnestedglist_lft[poseid][il]

            rhx.rbt.movearmfk(jnts_rgt, armname="rgt")
            rhx.rbt.movearmfk(jnts_lft, armname="lft")
            rhx.rbt.opengripper(jawwidth=jawwidth_rgt, armname="rgt")
            rhx.rbt.opengripper(jawwidth=jawwidth_lft, armname="lft")
            rhx.rbtmesh.genmnp(rhx.rbt).reparentTo(base.render)
            base.run()

