from commonimport import *

if __name__ == '__main__':

    yhx = robothelper.RobotHelperX(usereal=True)
    yhx.movetox(yhx.rbt.initrgtjnts, armname="rgt")
    yhx.movetox(yhx.rbt.initlftjnts, armname="lft")
    yhx.closegripperx(armname="rgt")
    yhx.closegripperx(armname="lft")
    yhx.opengripperx(armname="rgt")
    yhx.opengripperx(armname="lft")
    lctr = loc.Locator()

    armname = "rgt"
    ppplanner_tb = ppp.PickPlacePlanner(lctr.tubebigcm, yhx)
    ppplanner_ts = ppp.PickPlacePlanner(lctr.tubesmallcm, yhx)

    objpcd = lctr.capturecorrectedpcd(yhx.pxc)
    tshomomat = lctr.findtubestand_matchonobb(objpcd)
    elearray, eleconfidencearray = lctr.findtubes(tshomomat, objpcd, toggledebug=False)
    yhx.p3dh.genframe(pos=tshomomat[:3, 3], rotmat=tshomomat[:3, :3]).reparentTo(yhx.base.render)
    tubestandcm = lctr.gentubestand(homomat=tshomomat)

    tpobj = tp.TubePuzzle(elearray)
    if tpobj.isdone(tp.Node(elearray)):
        print("Tubes are arranged!")
        sys.exit()
    path = tpobj.atarSearch()
    # for state in path:
    #     print(state)
    #

    numikmsmpall = []
    jawwidthmsmpall = []
    objmsmpall = []
    othersmsmpall = []
    lastrgtarmjnts = yhx.rbt.initrgtjnts
    lastlftarmjnts = yhx.rbt.initlftjnts
    i = 0
    while i <= len(path) - 1:
        print(i, len(path) - 1)
        # try:
        nodepresent = path[i]
        nodenext = path[i + 1]
        griddiff = nodepresent.grid - nodenext.grid
        initij = np.where(griddiff > 0)
        initij = (initij[0][0], initij[1][0])
        tubeid = int(nodepresent.grid[initij[0], initij[1]])
        ppplanner = ppplanner_tb if tubeid == 1 else ppplanner_ts
        goalij = np.where(griddiff < 0)
        goalij = (goalij[0][0], goalij[1][0])
        collisionelearray = copy.deepcopy(nodepresent.grid)
        collisionelearray[initij[0], initij[1]] = 0
        renderingelearray = copy.deepcopy(nodepresent.grid)
        # renderingelearray[goalij[0], goalij[1]] = nodenext.grid[goalij[0], goalij[1]]
        # print(collisionelearray)
        # print(renderingelearray)
        print(nodepresent.grid)
        print(nodepresent.grid-nodenext.grid)
        collisiontbcmlist = lctr.gentubes(collisionelearray, tubestand_homomat=tshomomat)

        initpos_normalized = np.array(
            [lctr.tubeholecenters[initij[0], initij[1]][0], lctr.tubeholecenters[initij[0], initij[1]][1], 5])
        initpos = rm.homotransformpoint(tshomomat, initpos_normalized)
        inithm = copy.deepcopy(tshomomat)
        inithm[:3, 3] = initpos
        goalpos_normalized = np.array(
            [lctr.tubeholecenters[goalij[0], goalij[1]][0], lctr.tubeholecenters[goalij[0], goalij[1]][1], 5])
        goalpos = rm.homotransformpoint(tshomomat, goalpos_normalized)
        goalhm = copy.deepcopy(tshomomat)
        goalhm[:3, 3] = goalpos

        obscmlist = yhx.obscmlist + [tubestandcm] + collisiontbcmlist
        numikmsmp, jawwidthmsmp, objmsmp = ppplanner.findppmotion_symmetric(inithm, goalhm, armname=armname,
                                                                            rbtinitarmjnts = [lastrgtarmjnts, lastlftarmjnts],
                                                                            finalstate="uo",
                                                                            obscmlist=obscmlist, userrt=False)
        if numikmsmp is None:
            pass
        else:
            i += 1
        if armname is "rgt":
            lastrgtarmjnts = numikmsmp[-1][-1][1]
            lastlftarmjnts = lastlftarmjnts
        else:
            lastrgtarmjnts = lastrgtarmjnts
            lastlftarmjnts = numikmsmp[-1][-1][2]
        numikmsmpall += numikmsmp
        jawwidthmsmpall += jawwidthmsmp
        objcmmsmp = []
        for objhomomatmp in objmsmp:
            objcmmp = []
            for objhomomat in objhomomatmp:
                tmpobjcm = copy.deepcopy(ppplanner.objcm)
                tmpobjcm.set_homomat(objhomomat)
                if tubeid == 1:
                    tmpobjcm.setColor(1, 1, 0, 1)
                else:
                    tmpobjcm.setColor(1, 0, 1, 1)
                objcmmp.append(tmpobjcm)
            objcmmsmp.append(objcmmp)
        objmsmpall += objcmmsmp
        othersmsmp = []
        for idms in range(len(numikmsmp)):
            othersmp = []
            for idmp in range(len(numikmsmp[idms])):
                renderingtbcmlist = lctr.gentubes(renderingelearray, tubestand_homomat=tshomomat)
                othersmp.append([tubestandcm] + renderingtbcmlist)
            othersmsmp.append(othersmp)
        othersmsmpall += othersmsmp
    anime.animationgen(yhx, numikmsmpall, jawwidthmsmpall, objmsmpall, othersmsmpall)
    yhx.base.run()

    # counter = [0]
    # tubemnplist = [[]]
    # tubestandhomomat = [pos]
    # def update(path, counter, lctr, task):
    #     if counter[0] < len(path):
    #         lctr.showTubestand(pos=tubestandhomomat[0])
    #         state = path[counter[0]]
    #         lctr.showTubes(state.grid, tubestandhomomat[0])
    #         if base.inputmgr.keyMap['space'] is True:
    #             base.inputmgr.keyMap['space'] = False
    #             counter[0] += 1
    #     # else:
    #     #     counter[0] = 0
    #     return task.again
    #
    # taskMgr.doMethodLater(0.05, update, "update",
    #                       extraArgs=[path, counter, lctr],
    #                       appendTask=True)
    #
    # base.run()
