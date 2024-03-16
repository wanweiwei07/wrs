import copy
import numpy as np
import math

def animationgen(rhx, numikmsmp, jawwidthmsmp, objmsmp, othersmsmp, sg_doestimateagain=[False]):
    """
    create an animation with space

    :param base:
    :param robot_s: robot_sim/robot_s/robot_s.py
    :param rbtmg: robot_sim/robot_s/robotmesh.py
    :param numikmsmp: multisectional numikrms [waist, [left6], [right6]], ...
    :param jawwidthmsmp: multisectional ee_values [leftwidth, rightwidth], ...
    :param objmsmp: object multisectional motion sequence
    :param sg_doestimateagain: signal to control the external task
    :return:

    author: hao chen, revised by weiwei
    date: 20191122, 20200105
    """

    rbtmnpani = [None, None]
    objmnpani = [None]
    motionpathcounter = [0]
    objmpactive = [objmsmp[0]]
    numikmpactive = [numikmsmp[0]]
    jawwidthmpactive = [jawwidthmsmp[0]]
    othermpactive = [othersmsmp[0]]
    notherobjcms = 1000
    othersmnpani = [[None] * notherobjcms]  # define a sequence that is long enough, max 1000 objects]
    cleanuponlyflag = [False]

    def updatemotionpath(numikmpactive, jawwidthmpactive, objmpactive, othersmpactive, rbtmnp, objmnp, othersmnp,
                         motionpathcounter, rhx, cleanuponlyflag, task):
        if motionpathcounter[0] < len(numikmpactive[0]):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()
                # rbtmnp[1].detachNode()
            if objmnp[0] is not None:
                objmnp[0].detachNode()
            if othersmnp[0][0] is not None:
                for id, othermnp in enumerate(othersmnp[0]):
                    if othermnp is not None:
                        othermnp.detachNode()
                        othersmnp[0][id] = None
                    else:
                        break
            if cleanuponlyflag[0]:
                return task.done
            rgtarmjnts = numikmpactive[0][motionpathcounter[0]][1].tolist()
            lftarmjnts = numikmpactive[0][motionpathcounter[0]][2].tolist()
            rhx.robot_s.movealljnts([numikmpactive[0][motionpathcounter[0]][0], 0, 0] + rgtarmjnts + lftarmjnts)
            rgtjawwidth = jawwidthmpactive[0][motionpathcounter[0]][0]
            lftjawwidth = jawwidthmpactive[0][motionpathcounter[0]][1]
            # print rgtjawwidth, lftjawwidth
            rhx.robot_s.opengripper(armname='rgt', jawwidth=rgtjawwidth)
            rhx.robot_s.opengripper(armname='lft', jawwidth=lftjawwidth)
            rbtmnp[0] = rhx.rbtmesh.genmnp(rhx.robot_s)
            rbtmnp[0].reparentTo(rhx.base.render)
            objmnp[0] = objmpactive[0][motionpathcounter[0]]
            objmnp[0].reparentTo(rhx.base.render)
            objmnp[0].show_loc_frame()
            for idother, other in enumerate(othersmpactive[0][motionpathcounter[0]]):
                # tmpother = copy.copy(other)
                other.reparentTo(rhx.base.render)
                othersmnp[0][idother] = other
            motionpathcounter[0] += 1
            rhx.robot_s.goinitpose()
        else:
            motionpathcounter[0] = 0
            return task.again
        # base.win.saveScreenshot(Filename(str(motioncounter[0]) + '.jpg'))
        return task.again

    taskMgr.doMethodLater(0.01, updatemotionpath, "updatemotionpath",
                          extraArgs=[numikmpactive, jawwidthmpactive, objmpactive, othermpactive,
                                     rbtmnpani, objmnpani, othersmnpani, motionpathcounter, rhx, cleanuponlyflag],
                          appendTask=True)

    motionseccounter = [0]

    def updatemotionsection(numikmpactive, jawwidthmpactive, objmpactive, othermpactive,
                            numikmsmp, jawwidthmsmp, objmsmp, othersmsmp, rbtmnpani,
                            objmnpani, othersmnpani, motionpathcounter, motionseccounter, rhx, sg_doestimateagain,
                            task):
        if rhx.base.inputmgr.keymap['space'] is True:
            if motionseccounter[0] < len(objmsmp):
                motionseccounter[0] = motionseccounter[0] + 1
                if motionseccounter[0] < len(objmsmp):
                    objmpactive[0] = objmsmp[motionseccounter[0]]
                    numikmpactive[0] = numikmsmp[motionseccounter[0]]
                    jawwidthmpactive[0] = jawwidthmsmp[motionseccounter[0]]
                    othermpactive[0] = othersmsmp[motionseccounter[0]]
                    rhx.base.inputmgr.keymap['space'] = False
                    taskMgr.remove('updatemotionpath')
                    motionpathcounter[0] = 0
                    taskMgr.doMethodLater(0.03, updatemotionpath, "updatemotionpath",
                                          extraArgs=[numikmpactive, jawwidthmpactive, objmpactive, othermpactive,
                                                     rbtmnpani, objmnpani, othersmnpani, motionpathcounter, rhx,
                                                     [False]],
                                          appendTask=True)
                # execute
                # exeseccntr = motionseccounter[0] - 1
                # numikmp = numikmsmp[exeseccntr]
                # jawwidthmp = jawwidthmsmp[exeseccntr]
                # rgtjawwidth = jawwidthmp[0][0]
                # lftjawwidth = jawwidthmp[0][1]
                # if exeseccntr > 0:
                #     if not math.isclose(rgtjawwidth, jawwidthmsmp[exeseccntr - 1][0][0]):
                #         if rgtjawwidth < rhx.rgthndfa.jawwidthopen:
                #             rhx.closegripperx(arm_name="rgt")
                #         else:
                #             rhx.opengripperx(arm_name="rgt")
                #     if not math.isclose(lftjawwidth, jawwidthmsmp[exeseccntr - 1][0][1]):
                #         if lftjawwidth < rhx.lfthndfa.jawwidthopen:
                #             rhx.closegripperx(arm_name="lft")
                #         else:
                #             rhx.opengripperx(arm_name="lft")
                # if len(numikmp) != 1:
                #     arm_name = "rgt" if np.allclose(numikmp[0][2], numikmp[1][2]) else "lft"
                #     armid = 1 if arm_name is "rgt" else 2
                #     armjntspath = []
                #     for numik in numikmp:
                #         armjnts = numik[armid].tolist()
                #         armjntspath.append(armjnts)
                #     rhx.movemotionx(armjntspath, arm_name=arm_name)
            else:
                motionseccounter[0] = 0
                sg_doestimateagain[0] = True
                taskMgr.remove('updatemotionpath')
                # remove previous renderings
                taskMgr.add(updatemotionpath, "updatemotionpath",
                            extraArgs=[numikmpactive, jawwidthmpactive, objmpactive, othermpactive,
                                       rbtmnpani, objmnpani, othersmnpani, motionpathcounter, rhx, [True]],
                            appendTask=True)
                return task.done
        return task.again

    taskMgr.doMethodLater(0.04, updatemotionsection, "updatemotionsection",
                          extraArgs=[numikmpactive, jawwidthmpactive, objmpactive, othermpactive,
                                     numikmsmp, jawwidthmsmp, objmsmp, othersmsmp, rbtmnpani,
                                     objmnpani, othersmnpani, motionpathcounter, motionseccounter, rhx,
                                     sg_doestimateagain],
                          appendTask=True)