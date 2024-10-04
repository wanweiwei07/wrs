import numpy as np
import copy
import pickle
import os
import math
from utiltools.thirdparty import p3dhelper as p3dh

class PickPlacePlanner(object):
    def __init__(self, obj, rhx):
        """

        :param rbthi:
        :param hndovrmaster:
        :param obj:
        :param primitivedistance_foward:
        :param premitivedistance_backward:
        :param previous: [rgtarmj, lftarmj, rgthndwidth, lfthndwidth]

        author: hao chen, refactored by weiwei
        date: 20191122
        """

        if isinstance(obj, str):
            self.objname = obj
        elif isinstance(obj, rhx.mcm.CollisionModel):
            self.objname = obj.name
            self.objcm = obj

        self.rhx = rhx
        with open("./tro_predefinedsuctions.pickle", "rb") as file:
            graspdata = pickle.load(file)
            self.identityglist_rgt = graspdata[self.objname]
        with open("./tro_predefinedgrippings.pickle", "rb") as file:
            graspdata = pickle.load(file)
            self.identityglist_lft = graspdata[self.objname]

    def findsharedgrasps(self, inithomomat, goalhomomatlist, obscmlist, rbtarmjnts=None, armname="rgt",
                         toggledebug=False):
        """
        find a shared grasp, multiple goals are allowed
        the function finds the common grasp among the initmat and all rotmat in the goallist

        :param inithomomat:
        :param goalhomomatlist:
        :param obscmlist:
        :param rbtarmjnts: [lftarmjnts, rgtarmjnts], initarmjnts will be used in case of None
        :param armname:
        :param toggledebug:
        :return: a list of id pointing to the predefiend grasp list

        author: hao chen, refactored by weiwei
        date: 20191209, 20191212
        """

        objcmcopy = copy.deepcopy(self.objcm)

        rbt = self.rhx.robot_s
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        if rbtarmjnts is None:
            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
            rbt.movearmfk(rbt.initlftjnts, armname="lft")
        else:
            rbt.movearmfk(rbtarmjnts[0], armname="rgt")
            rbt.movearmfk(rbtarmjnts[1], armname="lft")
        rbtmg = self.rhx.rbtmesh
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        pcdchecker = self.rhx.pcdchecker
        bcdchecker = self.rhx.bcdchecker
        hndfa = self.rhx.rgthndfa
        p3dh = self.rhx.p3dh
        np = self.rhx.np
        rm = self.rhx.rm
        if armname is "lft":
            hndfa = self.rhx.lfthndfa
        # start pose
        objcmcopy.set_homomat(inithomomat)
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        availablegraspsatinit = []
        for idpre, predefined_grasp in enumerate(predefinedgrasps):
            predefined_jawwidth, predefined_fc, predefined_homomat = predefined_grasp
            hndmat4 = np.dot(inithomomat, predefined_homomat)
            eepos = rm.homotransformpoint(inithomomat, predefined_fc)[:3]
            eerot = hndmat4[:3, :3]
            hndnew = hndfa.genHand()
            hndnew.set_homomat(hndmat4)
            # hndnew.setjawwidth(predefined_jawwidth)
            hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
            ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
            if not ishndcollided:
                armjnts = rbt.numik(eepos, eerot, armname)
                if armjnts is not None:
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(0, 1, 0, .2)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
                    rbt.movearmfk(armjnts, armname)
                    isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                    # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], robot_s, arm_name, obscmlist)
                    isobjcollided = False  # for future use
                    if (not isrbtcollided) and (not isobjcollided):
                        if toggledebug:
                            rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                                self.rhx.base.render)
                        availablegraspsatinit.append(idpre)
                    elif (not isobjcollided):
                        if toggledebug:
                            rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[1, 0, 1, .5]).reparentTo(
                                self.rhx.base.render)
                else:
                    ikfailedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .6, 0, .2)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
                    # bcdchecker.showMeshList(hndnew.cmlist)
            else:
                ikcolliedgraspsnum += 1
                if toggledebug:
                    hndnew = hndfa.genHand()
                    hndnew.setColor(1, 0, 1, .2)
                    hndnew.set_homomat(hndmat4)
                    hndnew.setjawwidth(predefined_jawwidth)
                    hndnew.reparentTo(self.rhx.base.render)
                # bcdchecker.showMeshList(hndnew.cmlist)

        print("IK failed at the init pose: ", ikfailedgraspsnum)
        print("Collision at the init pose: ", ikcolliedgraspsnum)
        print("Possible number of the grasps at the init pose: ", len(availablegraspsatinit))

        # goal pose
        finalsharedgrasps = []
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        for index in range(len(goalhomomatlist)):
            objmat4 = goalhomomatlist[index]
            objcmcopy.set_homomat(objmat4)
            tmpresult = []
            for idavailableinit in availablegraspsatinit:
                predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idavailableinit]
                hndmat4 = np.dot(objmat4, predefined_homomat)
                eepos = rm.homotransformpoint(objmat4, predefined_fc)[:3]
                eerot = hndmat4[:3, :3]
                hndnew = hndfa.genHand()
                hndnew.set_homomat(hndmat4)
                # hndnew.setjawwidth(predefined_jawwidth)
                hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
                ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
                if not ishndcollided:
                    armjnts = rbt.numik(eepos, eerot, armname)
                    if armjnts is not None:
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(0, 1, 0, .5)
                            hndnew.set_homomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            hndnew.reparentTo(self.rhx.base.render)
                        rbt.movearmfk(armjnts, armname)
                        isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                        # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], robot_s, arm_name, obscmlist)
                        isobjcollided = False
                        if (not isrbtcollided) and (not isobjcollided):
                            tmpresult.append(idavailableinit)
                            if toggledebug:
                                rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False, rgbalft=[0, 1, 0, .5]).reparentTo(
                                    self.rhx.base.render)
                        elif (not isobjcollided):
                            if toggledebug:
                                rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False,
                                             rgbalft=[1, 0, 1, .5]).reparentTo(self.rhx.base.render)
                    else:
                        ikfailedgraspsnum += 1
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(1, .6, 0, .2)
                            hndnew.set_homomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            hndnew.reparentTo(self.rhx.base.render)
                else:
                    ikcolliedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, 0, 1, .2)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
            if index == 0:
                finalsharedgrasps = tmpresult
                availablegraspsatinit = finalsharedgrasps
            else:
                finalsharedgrasps = list(set(finalsharedgrasps).intersection(set(tmpresult)))
                availablegraspsatinit = finalsharedgrasps

        # if len(finalsharedgrasps) > 0:
        #     for idsharedgrasps in finalsharedgrasps:
        #         predefined_jawwidth, predefined_fc, predefined_homomat = predefined_graspss[idsharedgrasps]
        #         hndmat4 = np.dot(inithomomat, predefined_homomat)
        #         eepos = rm.homotransformpoint(inithomomat, predefined_fc)[:3]
        #         eerot = hndmat4[:3, :3]
        #         armjnts = robot_s.numik(eepos, eerot, arm_name)
        #         robot_s.movearmfk(armjnts, arm_name)
        #         rbtmg.genmnp(robot_s, togglejntscoord=False).reparentTo(self.rhx.base.render)

        print("IK failed grasps at the goal pose(s): ", ikfailedgraspsnum)
        print("Collision at the goal pose(s): ", ikcolliedgraspsnum)
        print("Possble number of shared grasps: ", len(finalsharedgrasps))

        rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
        rbt.movearmfk(bk_armjnts_lft, armname="lft")

        if len(finalsharedgrasps) == 0:
            return None
        else:
            return finalsharedgrasps

    def findsharedgrasps_symmetric(self, inithomomat, goalhomomatlist, obscmlist, symmetricaxis="z", nangles=9,
                                   rbtarmjnts=None, armname="rgt", toggledebug=False):
        """
        find a shared grasp, the goal is allowed to be symmetric

        :param inithomomat:
        :param goalhomomatlist:
        :param obscmlist:
        :param symmetricaxis:
        :param angleinterval:
        :param rbtarmjnts:
        :param armname:
        :param toggledebug:
        :return: [[goalhomomatlist], sharedgids]

        author: weiwei
        date: 20200107
        """

        objcmcopy = copy.deepcopy(self.objcm)

        rbt = self.rhx.robot_s
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        if rbtarmjnts is None:
            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
            rbt.movearmfk(rbt.initlftjnts, armname="lft")
        else:
            rbt.movearmfk(rbtarmjnts[0], armname="rgt")
            rbt.movearmfk(rbtarmjnts[1], armname="lft")
        rbtmg = self.rhx.rbtmesh
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        pcdchecker = self.rhx.pcdchecker
        bcdchecker = self.rhx.bcdchecker
        hndfa = self.rhx.rgthndfa
        p3dh = self.rhx.p3dh
        np = self.rhx.np
        rm = self.rhx.rm
        if armname is "lft":
            hndfa = self.rhx.lfthndfa
        # start pose
        objcmcopy.set_homomat(inithomomat)
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        availablegraspsatinit = []
        for idpre, predefined_grasp in enumerate(predefinedgrasps):
            predefined_jawwidth, predefined_fc, predefined_homomat = predefined_grasp
            hndmat4 = np.dot(inithomomat, predefined_homomat)
            eepos = rm.homotransformpoint(inithomomat, predefined_fc)[:3]
            eerot = hndmat4[:3, :3]
            hndnew = hndfa.genHand()
            hndnew.set_homomat(hndmat4)
            # hndnew.setjawwidth(predefined_jawwidth)
            hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
            ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
            if not ishndcollided:
                armjnts = rbt.numik(eepos, eerot, armname)
                if armjnts is not None:
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(0, 1, 0, .2)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
                    rbt.movearmfk(armjnts, armname)
                    isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                    # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], robot_s, arm_name, obscmlist)
                    isobjcollided = False  # for future use
                    if (not isrbtcollided) and (not isobjcollided):
                        if toggledebug:
                            rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                                self.rhx.base.render)
                        availablegraspsatinit.append(idpre)
                    elif (not isobjcollided):
                        if toggledebug:
                            rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[1, 0, 1, .5]).reparentTo(
                                self.rhx.base.render)
                else:
                    ikfailedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .6, 0, .2)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
                    # bcdchecker.showMeshList(hndnew.cmlist)
            else:
                ikcolliedgraspsnum += 1
                if toggledebug:
                    hndnew = hndfa.genHand()
                    hndnew.setColor(1, 0, 1, .2)
                    hndnew.set_homomat(hndmat4)
                    hndnew.setjawwidth(predefined_jawwidth)
                    hndnew.reparentTo(self.rhx.base.render)
                # bcdchecker.showMeshList(hndnew.cmlist)

        print("IK failed at the init pose: ", ikfailedgraspsnum)
        print("Collision at the init pose: ", ikcolliedgraspsnum)
        print("Possible number of the grasps at the init pose: ", len(availablegraspsatinit))

        # goal pose
        resultinghomomats = []
        finalsharedgrasps = []
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        for index, goalhm in enumerate(goalhomomatlist):
            if symmetricaxis is "z":
                rotax = goalhm[:3, 2]
            elif symmetricaxis is "y":
                rotax = goalhm[:3, 1]
            else:
                rotax = goalhm[:3, 0]
            candidateangles = np.linspace(0, 360, nangles)
            for rotangle in candidateangles:
                objmat4 = copy.deepcopy(goalhm)
                objmat4[:3, :3] = np.dot(rm.rodrigues(rotax, rotangle), goalhm[:3, :3])
                objcmcopy.set_homomat(objmat4)
                tmpresult = []
                for idavailableinit in availablegraspsatinit:
                    predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idavailableinit]
                    hndmat4 = np.dot(objmat4, predefined_homomat)
                    eepos = rm.homotransformpoint(objmat4, predefined_fc)[:3]
                    eerot = hndmat4[:3, :3]
                    hndnew = hndfa.genHand()
                    hndnew.set_homomat(hndmat4)
                    hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth,
                                                      jawwidthend=hndnew.jawwidthopen)
                    # hndnew.setjawwidth(predefined_jawwidth)
                    ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
                    if not ishndcollided:
                        armjnts = rbt.numik(eepos, eerot, armname)
                        if armjnts is not None:
                            if toggledebug:
                                hndnew = hndfa.genHand()
                                hndnew.setColor(0, 1, 0, .5)
                                hndnew.set_homomat(hndmat4)
                                hndnew.setjawwidth(predefined_jawwidth)
                                hndnew.reparentTo(self.rhx.base.render)
                            rbt.movearmfk(armjnts, armname)
                            isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                            # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], robot_s, arm_name, obscmlist)
                            isobjcollided = False
                            if (not isrbtcollided) and (not isobjcollided):
                                tmpresult.append(idavailableinit)
                                if toggledebug:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False, rgbalft=[0, 1, 0, .5]).reparentTo(
                                        self.rhx.base.render)
                            elif (not isobjcollided):
                                if toggledebug:
                                    rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False,
                                                 rgbalft=[1, 0, 1, .5]).reparentTo(self.rhx.base.render)
                        else:
                            ikfailedgraspsnum += 1
                            if toggledebug:
                                hndnew = hndfa.genHand()
                                hndnew.setColor(1, .6, 0, .2)
                                hndnew.set_homomat(hndmat4)
                                hndnew.setjawwidth(predefined_jawwidth)
                                hndnew.reparentTo(self.rhx.base.render)
                    else:
                        ikcolliedgraspsnum += 1
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(1, 0, 1, .2)
                            hndnew.set_homomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            hndnew.reparentTo(self.rhx.base.render)
                if len(tmpresult) == 0:
                    print("No shared grasps at symmetric angle " + str(rotangle) + ", trying the next...")
                    if rotangle == candidateangles[-1]:
                        print("No shared grasps!")
                        return [None, None]
                    continue
                else:
                    resultinghomomats.append(copy.deepcopy(objmat4))
                    if index == 0:
                        finalsharedgrasps = tmpresult
                        availablegraspsatinit = finalsharedgrasps
                    else:
                        finalsharedgrasps = list(set(finalsharedgrasps).intersection(set(tmpresult)))
                        availablegraspsatinit = finalsharedgrasps
                    break

        # if len(finalsharedgrasps) > 0:
        #     for idsharedgrasps in finalsharedgrasps:
        #         predefined_jawwidth, predefined_fc, predefined_homomat = predefined_graspss[idsharedgrasps]
        #         hndmat4 = np.dot(inithomomat, predefined_homomat)
        #         eepos = rm.homotransformpoint(inithomomat, predefined_fc)[:3]
        #         eerot = hndmat4[:3, :3]
        #         armjnts = robot_s.numik(eepos, eerot, arm_name)
        #         robot_s.movearmfk(armjnts, arm_name)
        #         rbtmg.genmnp(robot_s, togglejntscoord=False).reparentTo(self.rhx.base.render)

        print("IK failed grasps at the goal pose(s): ", ikfailedgraspsnum)
        print("Collision at the goal pose(s): ", ikcolliedgraspsnum)
        print("Possble number of shared grasps: ", len(finalsharedgrasps))

        rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
        rbt.movearmfk(bk_armjnts_lft, armname="lft")

        return [resultinghomomats, finalsharedgrasps]

    def pickup(self, objhomomat, grasp, primitivedirection_forward=np.array([0, 0, -1]),
               primitivedirection_backward=np.array([0, 0, 1]), primitivedistance_forward=100,
               primitivedistance_backward=100, armname="rgt", msc=None, obstaclecmlist=[]):
        """

        :param objhomomat:
        :param grasp: [prejawwidth, prehandcenter, prehandhomomat]
        :param retractdirection_forward:
        :param retractdirection_backward:
        :param retractdistance:
        :param armname:
        :param obstaclecmlist:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20200105
        """

        np = self.rhx.np
        rm = self.rhx.rm

        predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = grasp
        hndmat4 = np.dot(objhomomat, predefinedhandhomomat)
        eepos = rm.homotransformpoint(objhomomat, predefinedhndfc)
        eerot = hndmat4[:3, :3]
        if msc is None:
            amjnts = self.rhx.robot_s.numik(eepos, eerot, armname)
        else:
            amjnts = self.rhx.robot_s.numikmsc(eepos, eerot, msc, armname)
        if amjnts is not None:
            pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_forward, primitivedistance_forward,
                                                       amjnts, obstaclecmlist, armname)
            upmotion = self.rhx.genmovebackwardmotion(primitivedirection_backward, primitivedistance_backward,
                                                      amjnts, obstaclecmlist, armname)
            if pickmotion is not None and upmotion is not None:
                return [pickmotion, upmotion]
            else:
                if pickmotion is None:
                    print("pick error")
                if upmotion is None:
                    print("up error")
                return None
        else:
            print("no ik at the given grasp pose for pickup")
            return None

    def genppmotion(self, candidatepredgidlist, objinithomomat, objgoalhomomat, armname,
                    rbtinitarmjnts=None, rbtgoalarmjnts=None, finalstate="io",
                    primitivedirection_init_forward=None, primitivedirection_init_backward=None,
                    primitivedistance_init_foward=100, premitivedistance_init_backward=100,
                    primitivedirection_final_forward=None, primitivedirection_final_backward=None,
                    primitivedistance_final_foward=100, premitivedistance_final_backward=100, obscmlist=[],
                    userrt=True):
        """
        generate the pick and place motion for the speicified arm
        the robot_s init armjnts must be explicitly specified to avoid wrong robot_s poses

        :param candidatepredgidlist: candidate predefined grasp id list [int, int, ...]
        :param objinithomomat:
        :param objgoalhomomat:
        :param armname:
        :param rbtinitarmjnts, rbtgoalarmjnts: [lftarmjnts, rgtarmjnts], initial robot_s arm will be used if not set
        :param finalstate: use to orchestrate the motion, could be: "io"->backtoaninitialpose, grippers open, "uo"->backtoup, gripperopen, "gc"->stopatgoal, gripperclose
        :param primitivedirection_init_forward: the vector to move to the object init pose
        :param primitivedirection_init_backward: the vector to move back from the object init pose
        :param primitivedistance_init_forward
        :param primitivedistance_init_backward
        :param primitivedirection_final_forward: the vector to move to the object goal pose
        :param primitivedirection_final_backward: the vector to move back from the object goal pose
        :param primitivedistance_final_forward
        :param primitivedistance_final_backward
        :param userrt: rrt generator
        :param obscmlist:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20200105
        """

        if rbtinitarmjnts is not None and rbtgoalarmjnts is not None:
            if armname is "rgt":
                if not np.isclose(rbtinitarmjnts[0], rbtgoalarmjnts[0]):
                    print("The lft arm must maintain unmoved during generating ppsglmotion for the rgt arm.")
                    raise ValueError("")
            elif armname is "lft":
                if not np.isclose(rbtinitarmjnts[1], rbtgoalarmjnts[1]):
                    print("The rgt arm must maintain unmoved during generating ppsglmotion for the lft arm.")
                    raise ValueError("")
            else:
                print("Wrong arm_name. Must be rgt or lft.")
                raise ValueError("")

        rbt = self.rhx.robot_s
        hndfa = self.rhx.rgthndfa if armname is "rgt" else self.rhx.lfthndfa
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        bk_jawwidth_rgt = rbt.getjawwidth(armname="rgt")
        bk_jawwidth_lft = rbt.getjawwidth(armname="lft")
        if rbtinitarmjnts is None:
            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
            rbt.movearmfk(rbt.initlftjnts, armname="lft")
            rbt.opengripper(armname="rgt")
            rbt.opengripper(armname="lft")
        else:
            rbt.movearmfk(rbtinitarmjnts[0], armname="rgt")
            rbt.movearmfk(rbtinitarmjnts[1], armname="lft")
            rbt.opengripper(armname="rgt")
            rbt.opengripper(armname="lft")
        initpose = rbt.getarmjnts(armname=armname)
        if rbtgoalarmjnts is None:
            goalpose = rbt.initrgtjnts if armname is "rgt" else rbt.initlftjnts
        else:
            goalpose = rbtgoalarmjnts[1] if armname is "rgt" else rbtgoalarmjnts[0]

        if primitivedirection_init_forward is None:
            primitivedirection_init_forward = np.array([0, 0, -1])
        if primitivedirection_init_backward is None:
            primitivedirection_init_backward = np.array([0, 0, 1])
        if primitivedirection_final_forward is None:
            primitivedirection_final_forward = np.array([0, 0, -1])
        if primitivedirection_final_backward is None:
            primitivedirection_final_backward = np.array([0, 0, 1])

        for candidatepredgid in candidatepredgidlist:
            initpickup = self.pickup(objinithomomat, predefinedgrasps[candidatepredgid],
                                     primitivedirection_forward=primitivedirection_init_forward,
                                     primitivedirection_backward=primitivedirection_init_backward,
                                     primitivedistance_forward=primitivedistance_init_foward,
                                     primitivedistance_backward=premitivedistance_init_backward,
                                     armname=armname, obstaclecmlist=obscmlist)
            if initpickup is None:
                continue
            goalpickup = self.pickup(objgoalhomomat, predefinedgrasps[candidatepredgid],
                                     primitivedirection_forward=primitivedirection_final_forward,
                                     primitivedirection_backward=primitivedirection_final_backward,
                                     primitivedistance_forward=primitivedistance_final_foward,
                                     primitivedistance_backward=premitivedistance_final_backward,
                                     armname=armname, msc=initpickup[1][-1], obstaclecmlist=obscmlist)
            if goalpickup is None:
                continue
            initpick, initup = initpickup
            goalpick, goalup = goalpickup
            absinitpos = objinithomomat[:3, 3]
            absinitrot = objinithomomat[:3, :3]
            absgoalpos = objgoalhomomat[:3, 3]
            absgoalrot = objgoalhomomat[:3, :3]
            relpos, relrot = self.rhx.robot_s.getinhandpose(absinitpos, absinitrot, initpick[-1], armname)
            if userrt:
                rrtpathinit = self.rhx.planmotion(initpose, initpick[0], obscmlist, armname)
                if rrtpathinit is None:
                    print("No path found for moving from rbtinitarmjnts to pick up!")
                    continue
                rrtpathhold = self.rhx.planmotionhold(initup[-1], goalpick[0], self.objcm, relpos, relrot, obscmlist,
                                                      armname)
                if rrtpathhold is None:
                    print("No path found for moving from pick up to place down!")
                    continue
                if finalstate[0] is "i":
                    rrtpathgoal = self.rhx.planmotion(goalup[-1], goalpose, self.objcm, absgoalpos, absgoalrot,
                                                      obscmlist, armname)
                    if rrtpathgoal is None:
                        print("No path found for moving from place down to rbtgoalarmjnts!")
                        continue
            jawwidthopen = hndfa.jawwidthopen
            jawwidthclose = predefinedgrasps[candidatepredgid][0]
            break

        numikmsmp = []
        jawwidthmsmp = []
        objmsmp = []
        remainingarmjnts = rbt.getarmjnts(armname="rgt") if armname is "lft" else rbt.getarmjnts(armname="lft")
        if initpickup is not None and goalpickup is not None:
            if userrt:
                if rrtpathinit is not None and rrtpathhold is not None:
                    if finalstate[0] in ["i", "u"]:
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                              absinitpos, absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                              absgoalpos, absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen, absgoalpos,
                                                                              absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        if finalstate[0] is "i":
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathgoal, armname, jawwidthopen,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                    else:
                        finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                              rbt.getjawwidth(armname), absinitpos,
                                                                              absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                              absinitpos, absinitrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                  relpos, relrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, finaljawwidth,
                                                                              absgoalpos, absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                else:
                    print("There is possible sequence but no collision free motion!")
            else:
                if finalstate[0] in ["i", "u"]:
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                          absgoalpos, absgoalrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen, absgoalpos,
                                                                          absgoalrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    if finalstate[0] is "i":
                        numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpose], armname, jawwidthopen,
                                                                              absgoalpos, absgoalrot, remainingarmjnts)
                        numikmsmp.append(numikmp)
                        jawwidthmsmp.append(jawwidthmp)
                        objmsmp.append(objmp)
                else:
                    finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname, rbt.getjawwidth(armname),
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname, jawwidthclose,
                                                                          absinitpos, absinitrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose, relpos,
                                                                              relrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
                    numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, finaljawwidth,
                                                                          absgoalpos, absgoalrot, remainingarmjnts)
                    numikmsmp.append(numikmp)
                    jawwidthmsmp.append(jawwidthmp)
                    objmsmp.append(objmp)
            rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
            rbt.movearmfk(bk_armjnts_lft, armname="lft")
            rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
            rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
            return [numikmsmp, jawwidthmsmp, objmsmp]
        else:
            rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
            rbt.movearmfk(bk_armjnts_lft, armname="lft")
            rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
            rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
            return [None, None, None]

        print("No solution!")
        return [None, None, None]

    def findppmotion(self, objinithomomat, objgoalhomomat, obscmlist=[], symmetricaxis=None, nangles=9,
                       armname='rgt', rbtinitarmjnts=None, rbtgoalarmjnts=None, finalstate="io",
                       primitivedirection_init_forward=None, primitivedirection_init_backward=None,
                       primitivedistance_init_foward=150, premitivedistance_init_backward=150,
                       primitivedirection_final_forward=None, primitivedirection_final_backward=None,
                       primitivedistance_final_foward=150, premitivedistance_final_backward=150, userrt=True,
                       toggledebug=False):
        """
        this function performs findsharedgrasps and genppmotion in a loop

        :param candidatepredgidlist: candidate predefined grasp id list [int, int, ...]
        :param objinithomomat:
        :param objgoalhomomat:
        :param obscmlist:
        :param symmetricaxis: a special deal for symmetric objects, use None if not symmetric
        :param nangles: number of angles for the symmetricaxis, will be ignored if symmetricaxis is None
        :param armname:
        :param rbtinitarmjnts, rbtgoalarmjnts: [rgtsarmjnts, lftarmjnts], initial robot_s arm will be used if not set
        :param finalstate: use to orchestrate the motion, could be: "io"->backtoaninitialpose, grippers open, "uo"->backtoup, gripperopen, "gc"->stopatgoal, gripperclose
        :param primitivedirection_init_forward: the vector to move to the object init pose
        :param primitivedirection_init_backward: the vector to move back from the object init pose
        :param primitivedistance_init_forward
        :param primitivedistance_init_backward
        :param primitivedirection_final_forward: the vector to move to the object goal pose
        :param primitivedirection_final_backward: the vector to move back from the object goal pose
        :param primitivedistance_final_forward
        :param primitivedistance_final_backward
        :param userrt: rrt generator
        :param obscmlist:

        :return:

        author: weiwei
        date: 20200107
        """

        # find sharedgid

        objcmcopy = copy.deepcopy(self.objcm)

        rbt = self.rhx.robot_s
        bk_armjnts_rgt = rbt.getarmjnts(armname="rgt")
        bk_armjnts_lft = rbt.getarmjnts(armname="lft")
        bk_jawwidth_rgt = rbt.getjawwidth(armname="rgt")
        bk_jawwidth_lft = rbt.getjawwidth(armname="lft")
        rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
        rbt.movearmfk(rbt.initlftjnts, armname="lft")
        rbt.opengripper(armname="rgt")
        rbt.opengripper(armname="lft")
        rbtmg = self.rhx.rbtmesh
        hndfa = self.rhx.rgthndfa if armname is "rgt" else self.rhx.lfthndfa
        predefinedgrasps = self.identityglist_rgt if armname is "rgt" else self.identityglist_lft
        pcdchecker = self.rhx.pcdchecker
        bcdchecker = self.rhx.bcdchecker
        np = self.rhx.np
        rm = self.rhx.rm
        # start pose
        objcmcopy.set_homomat(objinithomomat)
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0
        availablegraspsatinit = []
        for idpre, predefined_grasp in enumerate(predefinedgrasps):
            # if toggle_dbg:
            #     availablegraspsatinit.append(idpre)
            predefined_jawwidth, predefined_fc, predefined_homomat = predefined_grasp
            hndmat4 = np.dot(objinithomomat, predefined_homomat)
            eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
            eerot = hndmat4[:3, :3]
            hndnew = hndfa.genHand()
            hndnew.set_homomat(hndmat4)
            # hndnew.setjawwidth(predefined_jawwidth)
            hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
            ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
            if not ishndcollided:
                armjnts = rbt.numik(eepos, eerot, armname)
                if armjnts is not None:
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(0, 1, 0, .2)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
                    rbt.movearmfk(armjnts, armname)
                    isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                    # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], robot_s, arm_name, obscmlist)
                    isobjcollided = False  # for future use
                    if (not isrbtcollided) and (not isobjcollided):
                        if toggledebug:
                            rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                                self.rhx.base.render)
                            pass
                        # if not toggle_dbg:
                        #     availablegraspsatinit.append(idpre)
                        availablegraspsatinit.append(idpre)
                    elif (not isobjcollided):
                        if toggledebug:
                            rbtmg.genmnp(rbt, drawhand=False, togglejntscoord=False, toggleendcoord=False, rgbargt=[1, 0, 1, .5]).reparentTo(
                                self.rhx.base.render)
                            pass
                else:
                    ikfailedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .6, 0, .7)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        hndnew.reparentTo(self.rhx.base.render)
                    # bcdchecker.showMeshList(hndnew.cmlist)
            else:
                ikcolliedgraspsnum += 1
                if toggledebug:
                    hndnew = hndfa.genHand()
                    hndnew.setColor(1, .5, .5, .7)
                    hndnew.set_homomat(hndmat4)
                    hndnew.setjawwidth(predefined_jawwidth)
                    hndnew.reparentTo(self.rhx.base.render)
                # bcdchecker.showMeshList(hndnew.cmlist)

        print("IK failed at the init pose: ", ikfailedgraspsnum)
        print("Collision at the init pose: ", ikcolliedgraspsnum)
        print("Possible number of the grasps at the init pose: ", len(availablegraspsatinit))

        if toggledebug:
            base.run()
        # goal pose
        finalsharedgrasps = []
        ikfailedgraspsnum = 0
        ikcolliedgraspsnum = 0

        if symmetricaxis is None:
            candidateangles = [0]
            rotax = objgoalhomomat[:3, 0]
        else:
            if symmetricaxis is "z":
                rotax = objgoalhomomat[:3, 2]
            elif symmetricaxis is "y":
                rotax = objgoalhomomat[:3, 1]
            else:
                rotax = objgoalhomomat[:3, 0]
            candidateangles = np.linspace(0, 360, nangles)
        if toggledebug:
            candidateangles = [90.0]
        for rotangle in candidateangles:
            objmat4 = copy.deepcopy(objgoalhomomat)
            objmat4[:3, :3] = np.dot(rm.rodrigues(rotax, rotangle), objgoalhomomat[:3, :3])
            objcmcopy.set_homomat(objmat4)
            for idavailableinit in availablegraspsatinit:
                predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idavailableinit]
                hndmat4 = np.dot(objmat4, predefined_homomat)
                eepos = rm.homotransformpoint(objmat4, predefined_fc)[:3]
                eerot = hndmat4[:3, :3]
                hndnew = hndfa.genHand()
                hndnew.set_homomat(hndmat4)
                hndcmlist = hndnew.genrangecmlist(jawwidthstart=predefined_jawwidth, jawwidthend=hndnew.jawwidthopen)
                # hndnew.setjawwidth(predefined_jawwidth)
                ishndcollided = bcdchecker.isMeshListMeshListCollided(hndcmlist, obscmlist)
                if not ishndcollided:
                    armjnts = rbt.numik(eepos, eerot, armname)
                    if armjnts is not None:
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(0, 1, 0, .5)
                            hndnew.set_homomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            # hndnew.reparentTo(self.rhx.base.render)
                        rbt.movearmfk(armjnts, armname)
                        isrbtcollided = pcdchecker.isRobotCollided(rbt, obscmlist, holdarmname=armname)
                        # isobjcollided = pcdchecker.isObjectsOthersCollided([obj], robot_s, arm_name, obscmlist)
                        isobjcollided = False
                        if (not isrbtcollided) and (not isobjcollided):
                            finalsharedgrasps.append(idavailableinit)
                            if toggledebug:
                                rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                             rgbargt=[0, 1, 0, .5]).reparentTo(
                                    self.rhx.base.render)
                                # rbtmg.genmnp(robot_s, togglejntscoord=False).reparentTo(self.rhx.base.render)
                                # # toggle the following one in case both the common start and goal shall be rendered
                                # hndmat4 = np.dot(objinithomomat, predefined_homomat)
                                # # hndnew = hndfa.genHand()
                                # # hndnew.setColor(0, 1, 0, .5)
                                # # hndnew.sethomomat(hndmat4)
                                # # hndnew.setjawwidth(predefined_jawwidth)
                                # # hndnew.reparentTo(self.rhx.base.render)
                                # eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
                                # eerot = hndmat4[:3, :3]
                                # armjnts = robot_s.numik(eepos, eerot, arm_name)
                                # robot_s.movearmfk(armjnts, arm_name)
                                # rbtmg.genmnp(robot_s, togglejntscoord=False, toggleendcoord=False, rgbargt=[0, 1, 0, .5]).reparentTo(
                                #     self.rhx.base.render)
                                # # base.run()
                                # rbtmg.genmnp(robot_s, togglejntscoord=False).reparentTo(self.rhx.base.render)
                                pass
                        elif (not isobjcollided):
                            if toggledebug:
                                # rbtmg.genmnp(robot_s, drawhand=False, togglejntscoord=False, toggleendcoord=False,
                                #              rgbargt=[1, 0, 1, .5]).reparentTo(self.rhx.base.render)
                                pass
                    else:
                        ikfailedgraspsnum += 1
                        if toggledebug:
                            hndnew = hndfa.genHand()
                            hndnew.setColor(1, .6, 0, .7)
                            hndnew.set_homomat(hndmat4)
                            hndnew.setjawwidth(predefined_jawwidth)
                            # hndnew.reparentTo(self.rhx.base.render)
                else:
                    ikcolliedgraspsnum += 1
                    if toggledebug:
                        hndnew = hndfa.genHand()
                        hndnew.setColor(1, .5, .5, .7)
                        hndnew.set_homomat(hndmat4)
                        hndnew.setjawwidth(predefined_jawwidth)
                        # hndnew.reparentTo(self.rhx.base.render)
            print("IK failed grasps at the goal pose: ", ikfailedgraspsnum)
            print("Collision at the goal pose: ", ikcolliedgraspsnum)
            print("Possble number of shared grasps: ", len(finalsharedgrasps))
            if toggledebug:
                base.run()

            if len(finalsharedgrasps) == 0:
                print("No shared grasps at symmetric angle " + str(rotangle) + ", trying the next...")
                if rotangle == candidateangles[-1]:
                    print("No shared grasps!")
                    return [None, None, None]
                continue
            else:
                resultinghomomat = copy.deepcopy(objmat4)
                resultinghomomat[:3, 3] = resultinghomomat[:3, 3] + resultinghomomat[:3,
                                                                    2] * 5  # move 5mm up, do not move until end_type
                # if toggle_dbg:
                #     for idsharedgrasps in finalsharedgrasps:
                #         predefined_jawwidth, predefined_fc, predefined_homomat = predefinedgrasps[idsharedgrasps]
                #         hndmat4 = np.dot(objinithomomat, predefined_homomat)
                #         eepos = rm.homotransformpoint(objinithomomat, predefined_fc)[:3]
                #         eerot = hndmat4[:3, :3]
                #         armjnts = robot_s.numik(eepos, eerot, arm_name)
                #         robot_s.movearmfk(armjnts, arm_name)
                #         rbtmg.genmnp(robot_s, togglejntscoord=False).reparentTo(self.rhx.base.render)

                if rbtinitarmjnts is not None and rbtgoalarmjnts is not None:
                    if armname is "rgt":
                        if not np.isclose(rbtinitarmjnts[1], rbtgoalarmjnts[1]):
                            print("The lft arm must maintain unmoved during generating ppsglmotion for the rgt arm.")
                            raise ValueError("")
                    elif armname is "lft":
                        if not np.isclose(rbtinitarmjnts[0], rbtgoalarmjnts[0]):
                            print("The rgt arm must maintain unmoved during generating ppsglmotion for the lft arm.")
                            raise ValueError("")
                    else:
                        print("Wrong arm_name. Must be rgt or lft.")
                        raise ValueError("")
                if rbtinitarmjnts is not None:
                    rbt.movearmfk(rbtinitarmjnts[0], armname="rgt")
                    rbt.movearmfk(rbtinitarmjnts[1], armname="lft")
                    rbt.opengripper(armname="rgt")
                    rbt.opengripper(armname="lft")
                else:
                    rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                    rbt.movearmfk(rbt.initlftjnts, armname="lft")
                    rbt.opengripper(armname="rgt")
                    rbt.opengripper(armname="lft")
                initpose = rbt.getarmjnts(armname=armname)
                if rbtgoalarmjnts is None:
                    goalpose = rbt.initrgtjnts if armname is "rgt" else rbt.initlftjnts
                else:
                    goalpose = rbtgoalarmjnts[0] if armname is "rgt" else rbtgoalarmjnts[1]

                if primitivedirection_init_forward is None:
                    primitivedirection_init_forward = np.array([0, 0, -1])
                    # primitivedirection_init_forward = -objinithomomat[:3, 2]
                if primitivedirection_init_backward is None:
                    primitivedirection_init_backward = np.array([0, 0, 1])
                    # primitivedirection_init_backward = objinithomomat[:3, 2]
                if primitivedirection_final_forward is None:
                    primitivedirection_final_forward = np.array([0, 0, -1])
                    # primitivedirection_final_forward = -objgoalhomomat[:3, 2]
                if primitivedirection_final_backward is None:
                    primitivedirection_final_backward = np.array([0, 0, 1])
                    # primitivedirection_final_backward = objgoalhomomat[:3, 2]

                initpickup = None
                goalpickup = None
                for candidatepredgid in finalsharedgrasps:
                    print("picking up...")
                    initpickup = self.pickup(objinithomomat, predefinedgrasps[candidatepredgid],
                                             primitivedirection_forward=primitivedirection_init_forward,
                                             primitivedirection_backward=primitivedirection_init_backward,
                                             primitivedistance_forward=primitivedistance_init_foward,
                                             primitivedistance_backward=premitivedistance_init_backward,
                                             armname=armname, obstaclecmlist=obscmlist)
                    if initpickup is None:
                        if toggledebug and rotangle == candidateangles[-1]:
                            predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = predefinedgrasps[
                                candidatepredgid]
                            hndmat4 = np.dot(objinithomomat, predefinedhandhomomat)
                            eepos = rm.homotransformpoint(objinithomomat, predefinedhndfc)
                            eerot = hndmat4[:3, :3]
                            amjnts = self.rhx.robot_s.numik(eepos, eerot, armname)
                            rbt.movearmfk(armjnts, armname="rgt")
                            if amjnts is not None:
                                pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_init_forward,
                                                                           primitivedistance_init_foward,
                                                                           amjnts, armname)
                                if pickmotion is None:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([1, 0, 0, .7])).reparentTo(self.rhx.base.render)
                                    if candidatepredgid == finalsharedgrasps[-1]:
                                        base.run()
                            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                            rbt.movearmfk(rbt.initlftjnts, armname="lft")
                        continue
                    print("placing down...")
                    goalpickup = self.pickup(resultinghomomat, predefinedgrasps[candidatepredgid],
                                             primitivedirection_forward=primitivedirection_final_forward,
                                             primitivedirection_backward=primitivedirection_final_backward,
                                             primitivedistance_forward=primitivedistance_final_foward,
                                             primitivedistance_backward=premitivedistance_final_backward,
                                             armname=armname, msc=initpickup[1][-1], obstaclecmlist=obscmlist)
                    if goalpickup is None:
                        if toggledebug and rotangle == candidateangles[-1]:
                            predefinedjawwidth, predefinedhndfc, predefinedhandhomomat = predefinedgrasps[
                                candidatepredgid]
                            hndmat4 = np.dot(objinithomomat, predefinedhandhomomat)
                            eepos = rm.homotransformpoint(objinithomomat, predefinedhndfc)
                            eerot = hndmat4[:3, :3]
                            amjnts = self.rhx.robot_s.numikmsc(eepos, eerot, initpickup[1][-1], armname)
                            rbt.movearmfk(armjnts, armname="rgt")
                            if amjnts is not None:
                                pickmotion = self.rhx.genmoveforwardmotion(primitivedirection_init_forward,
                                                                           primitivedistance_init_foward,
                                                                           amjnts, armname)
                                if pickmotion is None:
                                    # rbtmg.genmnp(robot_s, togglejntscoord=False, toggleendcoord=False, rgbargt=np.array([1, 0, 0, .7])).reparentTo(self.rhx.base.render)
                                    if candidatepredgid == finalsharedgrasps[-1]:
                                        base.run()
                            rbt.movearmfk(rbt.initrgtjnts, armname="rgt")
                            rbt.movearmfk(rbt.initlftjnts, armname="lft")
                        continue
                    initpick, initup = initpickup
                    goalpick, goalup = goalpickup
                    absinitpos = objinithomomat[:3, 3]
                    absinitrot = objinithomomat[:3, :3]
                    absgoalpos = resultinghomomat[:3, 3]
                    absgoalrot = resultinghomomat[:3, :3]
                    relpos, relrot = self.rhx.robot_s.getinhandpose(absinitpos, absinitrot, initpick[-1], armname)
                    if userrt:
                        rrtpathinit = self.rhx.planmotion(initpose, initpick[0], obscmlist, armname)
                        if rrtpathinit is None:
                            print("No path found for moving from rbtinitarmjnts to pick up!")
                            continue
                        rrtpathhold = self.rhx.planmotionhold(initup[-1], goalpick[0], self.objcm, relpos, relrot,
                                                              obscmlist, armname)
                        if rrtpathhold is None:
                            print("No path found for moving from pick up to place down!")
                            continue
                        if finalstate[0] is "i":
                            rrtpathgoal = self.rhx.planmotion(goalup[-1], goalpose, obscmlist, armname)
                            if rrtpathgoal is None:
                                print("No path found for moving from place down to rbtgoalarmjnts!")
                                continue
                    jawwidthopen = hndfa.jawwidthopen
                    jawwidthclose = predefinedgrasps[candidatepredgid][0]
                    break

                numikmsmp = []
                jawwidthmsmp = []
                objmsmp = []
                remainingarmjnts = rbt.getarmjnts(armname="rgt") if armname is "lft" else rbt.getarmjnts(armname="lft")
                if initpickup is not None and goalpickup is not None:
                    if userrt:
                        if rrtpathinit is not None and rrtpathhold is not None:
                            if finalstate[0] in ["i", "u"]:
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                      jawwidthclose,
                                                                                      absinitpos, absinitrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                      jawwidthopen,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen,
                                                                                      absgoalpos,
                                                                                      absgoalrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                if finalstate[0] is "i":
                                    numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathgoal, armname,
                                                                                          jawwidthopen,
                                                                                          absgoalpos, absgoalrot,
                                                                                          remainingarmjnts)
                                    numikmsmp.append(numikmp)
                                    jawwidthmsmp.append(jawwidthmp)
                                    objmsmp.append(objmp)
                            else:
                                finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(rrtpathinit, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                      rbt.getjawwidth(armname),
                                                                                      absinitpos,
                                                                                      absinitrot, remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                      jawwidthclose,
                                                                                      absinitpos, absinitrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(rrtpathhold, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname,
                                                                                          jawwidthclose,
                                                                                          relpos, relrot,
                                                                                          remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                      finaljawwidth,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                        else:
                            print("There is possible sequence but no collision free motion!")
                    else:
                        if finalstate[0] in ["i", "u"]:
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                  jawwidthclose,
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname, jawwidthopen,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(goalup, armname, jawwidthopen,
                                                                                  absgoalpos,
                                                                                  absgoalrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            if finalstate[0] is "i":
                                numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpose], armname, jawwidthopen,
                                                                                      absgoalpos, absgoalrot,
                                                                                      remainingarmjnts)
                                numikmsmp.append(numikmp)
                                jawwidthmsmp.append(jawwidthmp)
                                objmsmp.append(objmp)
                        else:
                            finaljawwidth = jawwidthopen if finalstate[1] is "o" else jawwidthclose
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpose], armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath(initpick, armname,
                                                                                  rbt.getjawwidth(armname),
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([initpick[-1]], armname,
                                                                                  jawwidthclose,
                                                                                  absinitpos, absinitrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(initup, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpathhold(goalpick, armname, jawwidthclose,
                                                                                      relpos,
                                                                                      relrot, remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                            numikmp, jawwidthmp, objmp = self.formulatemotionpath([goalpick[-1]], armname,
                                                                                  finaljawwidth,
                                                                                  absgoalpos, absgoalrot,
                                                                                  remainingarmjnts)
                            numikmsmp.append(numikmp)
                            jawwidthmsmp.append(jawwidthmp)
                            objmsmp.append(objmp)
                    rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
                    rbt.movearmfk(bk_armjnts_lft, armname="lft")
                    rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
                    rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)

                    if toggledebug:
                        # draw motion
                        for i, numikms in enumerate(numikmsmp):
                            for j, numik in enumerate(numikms):
                                rbt.movearmfk(numik[1], armname="rgt")
                                rbt.movearmfk(numik[2], armname="lft")
                                rbt.opengripper(armname="rgt", jawwidth=jawwidthmsmp[i][j][0])
                                rbt.opengripper(armname="lft", jawwidth=jawwidthmsmp[i][j][1])
                                if j == 0 or j == len(numikms):
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([0, 1, 0, 1])).reparentTo(self.rhx.base.render)
                                else:
                                    rbtmg.genmnp(rbt, togglejntscoord=False, toggleendcoord=False,
                                                 rgbargt=np.array([.5, .5, .5, .5])).reparentTo(self.rhx.base.render)
                                if j < len(numikms) - 1:
                                    numiknxt = numikms[j + 1][1]
                                    thispos, _ = rbt.getee(armname="rgt")
                                    rbt.movearmfk(numiknxt, armname="rgt")
                                    nxtpos, _ = rbt.getee(armname="rgt")
                                    p3dh.genlinesegnodepath([thispos, nxtpos], colors=[1, 0, 1, 1],
                                                            thickness=3.7).reparentTo(base.render)
                                objcp = copy.deepcopy(self.objcm)
                                objcp.set_homomat(objmsmp[i][j])
                                objcp.setColor(1, 1, 0, 1)
                                objcp.reparentTo(base.render)
                        base.run()

                    return [numikmsmp, jawwidthmsmp, objmsmp]
                else:
                    rbt.movearmfk(bk_armjnts_rgt, armname="rgt")
                    rbt.movearmfk(bk_armjnts_lft, armname="lft")
                    rbt.opengripper(armname="rgt", jawwidth=bk_jawwidth_rgt)
                    rbt.opengripper(armname="lft", jawwidth=bk_jawwidth_lft)
                    if initpickup is None:
                        print("No feasible motion for pick up!")
                    if goalpickup is None:
                        print("No feasible motion for placing down!")
                    print("The shared grasps failed at symmetric angle " + str(rotangle) + ", trying the next...")
                    continue

        # if toggle_dbg:
        #     base.run()
        print("No feasible motion between two key poses!")
        return [None, None, None]

    def formulatemotionpath(self, motionpath, armname, jawwidth, objpos, objrot, remainingarmjnts):
        """

        :param motionpath:
        :param jawwidth:
        :param relpos:
        :param relrot:
        :param remainingarmjnts
        :param armname:
        :return:

        author: weiwei
        date: 20200105
        """

        numikmp = []
        jawwidthmp = []
        objmp = []

        rbt = self.rhx.robot_s
        rm = self.rhx.rm

        for elemp in motionpath:
            if armname is "rgt":
                lftelemp = remainingarmjnts
                rgtelemp = np.asarray(elemp)
                lftelejw = rbt.getjawwidth("lft")
                rgtelejw = jawwidth
            elif armname is "lft":
                lftelemp = np.asarray(elemp)
                rgtelemp = remainingarmjnts
                lftelejw = jawwidth
                rgtelejw = rbt.getjawwidth("rgt")
            numikmp.append(np.array([0, rgtelemp, lftelemp]))
            jawwidthmp.append([rgtelejw, lftelejw])
            objmp.append(rm.homobuild(objpos, objrot))

        return [numikmp, jawwidthmp, objmp]

    def formulatemotionpathhold(self, motionpath, armname, jawwidth, relpos, relrot, remainingarmjnts):
        """

        :param motionpath:
        :param jawwidth:
        :param relpos:
        :param relrot:
        :param remainingarmjnts
        :param armname:
        :return:

        author: weiwei
        date: 20200105
        """

        numikmp = []
        jawwidthmp = []
        objmp = []

        rbt = self.rhx.robot_s
        rm = self.rhx.rm
        bk_armjnts = rbt.getarmjnts(armname=armname)

        for elemp in motionpath:
            if armname is "rgt":
                lftelemp = remainingarmjnts
                rgtelemp = np.asarray(elemp)
                lftelejw = rbt.getjawwidth("lft")
                rgtelejw = jawwidth
            elif armname is "lft":
                rgtelemp = remainingarmjnts
                lftelemp = np.asarray(elemp)
                lftelejw = jawwidth
                rgtelejw = rbt.getjawwidth("rgt")
            numikmp.append(np.array([0, rgtelemp, lftelemp]))
            jawwidthmp.append([rgtelejw, lftelejw])
            rbt.movearmfk(elemp, armname=armname)
            objpos, objrot = rbt.getworldpose(relpos, relrot, armname=armname)
            objmp.append(rm.homobuild(objpos, objrot))
        rbt.movearmfk(bk_armjnts, armname=armname)

        return [numikmp, jawwidthmp, objmp]
