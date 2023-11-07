from commonimport import *

if __name__ == '__main__':

    yhx = robothelper.RobotHelperX(usereal=True)
    yhx.movetox(yhx.rbt.initrgtjnts, armname="rgt")
    yhx.movetox(yhx.rbt.initlftjnts, armname="lft")
    yhx.closegripperx(armname="rgt")
    yhx.closegripperx(armname="lft")
    yhx.opengripperx(armname="rgt")
    yhx.opengripperx(armname="lft")
    lctr = locfixed.LocatorFixed(homomatfilename_start="rightfixture_light_homomat1", homomatfilename_goal="rightfixture_light_homomat2")

    armname = "rgt"
    ppplanner_tb = ppp.PickPlacePlanner(lctr.tubebigcm, yhx)
    ppplanner_ts = ppp.PickPlacePlanner(lctr.tubesmallcm, yhx)

    objpcd = lctr.capturecorrectedpcd(yhx.pxc)
    elearray, eleconfidencearray = lctr.findtubes(lctr.tubestandhomomat_start, objpcd, toggledebug=False)
    yhx.p3dh.genframe(pos=lctr.tubestandhomomat_start[:3, 3], rotmat=lctr.tubestandhomomat_start[:3, :3]).reparentTo(yhx.base.render)
    tubestandcmlist = [lctr.gentubestand(homomat=lctr.tubestandhomomat_start), lctr.gentubestand(homomat=lctr.tubestandhomomat_goal, rgba = np.array([0.1, .5, .7, 1.]))]

    nrow = elearray.shape[0]
    ncol = elearray.shape[1]
    # expand state
    elearray_ext = np.zeros((nrow*2, ncol))
    elearray_ext[:nrow, :] = elearray[:, :]

    tpobj = tp.TubePuzzle(elearray_ext)
    if tpobj.isdone(tp.Node(elearray_ext)):
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
    # for bad state
    badarraylist = []
    badstartgoals = []
    while i < len(path) - 1:
        print(i, len(path) - 1)
        # try:
        nodepresent = path[i]
        nodenext = path[i + 1]
        griddiff = nodepresent.grid - nodenext.grid
        initij = np.where(griddiff > 0)
        initij = (initij[0][0], initij[1][0])
        tubeid = int(nodepresent.grid[initij[0], initij[1]])
        ppplanner = ppplanner_tb if tubeid == 1 else ppplanner_ts
        goalijglobal = np.where(griddiff < 0)
        if goalijglobal[0][0] < nrow:
            goalijlocal = (goalijglobal[0][0], goalijglobal[1][0])
            tubestandhomomat_goal = lctr.tubestandhomomat_start
        else:
            goalijlocal = (goalijglobal[0][0]-nrow, goalijglobal[1][0])
            tubestandhomomat_goal = lctr.tubestandhomomat_goal
        collisionelearray = copy.deepcopy(nodepresent.grid)
        collisionelearray[initij[0], initij[1]] = 0
        renderingelearray = copy.deepcopy(nodepresent.grid)
        # renderingelearray[goalijglobal[0], goalijglobal[1]] = nodenext.grid[goalijglobal[0], goalijglobal[1]]
        # print(collisionelearray)
        # print(renderingelearray)
        print(nodepresent.grid)
        print(nodepresent.grid-nodenext.grid)
        collisiontbcmlist = lctr.gentubes(collisionelearray[:nrow, :], tubestand_homomat=lctr.tubestandhomomat_start)
        # collisiontbcmlist = [lctr.gentubeandstandboxcm(pos=lctr.tubestandhomomat_start)]
        # collisiontbcmlist[0].reparentTo(yhx.base.render)
        collisiontbcmlist += lctr.gentubes(collisionelearray[nrow:, :], tubestand_homomat=lctr.tubestandhomomat_goal)

        initpos_normalized = np.array(
            [lctr.tubeholecenters[initij[0], initij[1]][0], lctr.tubeholecenters[initij[0], initij[1]][1], 5])
        initpos = rm.homotransformpoint(lctr.tubestandhomomat_start, initpos_normalized)
        inithm = copy.deepcopy(lctr.tubestandhomomat_start)
        inithm[:3, 3] = initpos
        goalpos_normalized = np.array(
            [lctr.tubeholecenters[goalijlocal[0], goalijlocal[1]][0], lctr.tubeholecenters[goalijlocal[0], goalijlocal[1]][1], 10])
        goalpos = rm.homotransformpoint(tubestandhomomat_goal, goalpos_normalized)
        goalhm = copy.deepcopy(tubestandhomomat_goal)
        goalhm[:3, 3] = goalpos

        obscmlist = yhx.obscmlist + tubestandcmlist + collisiontbcmlist
        numikmsmp, jawwidthmsmp, objmsmp = ppplanner.findppmotion_symmetric(inithm, goalhm, armname=armname,
                                                                            rbtinitarmjnts = [lastrgtarmjnts, lastlftarmjnts],
                                                                            finalstate="uo", obscmlist=obscmlist,
                                                                            nangles=5, userrt=True,
                                                                            primitivedistance_init_foward=150,
                                                                            premitivedistance_init_backward=150,
                                                                            primitivedistance_final_foward=150,
                                                                            premitivedistance_final_backward=150,
                                                                            toggledebug = False)
        ## for toggle_dbg, check the collisions between the hand and the tubes
        # for tbcm in collisiontbcmlist:
        #     tbcm.reparentTo(yhx.base.render)
        #     tbcm.setColor(1,0,0,.2)
        #     tbcm.showcn()
        # yhx.base.run()
        if numikmsmp is None:
            tmpelearray = copy.deepcopy(nodepresent.grid)
            badarraylist.append(tmpelearray)
            badstartgoals.append([[initij[0], initij[1]], [goalijglobal[0], goalijglobal[1]]])
            tpobj = tp.TubePuzzle(tmpelearray)
            weightarray = np.zeros_like(nodepresent.grid)
            for id, onebadarray in enumerate(badarraylist):
                if np.array_equal(onebadarray, tmpelearray):
                    weightarray[badstartgoals[id][0][0], badstartgoals[id][0][1]] = id+1
                    weightarray[badstartgoals[id][1][0], badstartgoals[id][1][1]] = id+1
            path = tpobj.atarSearch(weightarray)
            print("###weight_array", weightarray)
            i = 0
            # print("Failure")
            # print(badarraylist)
            # print(weight_array)
            continue
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
                renderingtbcmlist = lctr.gentubes(renderingelearray[:nrow, :], tubestand_homomat=lctr.tubestandhomomat_start)
                renderingtbcmlist += lctr.gentubes(renderingelearray[nrow:, :], tubestand_homomat=lctr.tubestandhomomat_goal)
                othersmp.append(tubestandcmlist + renderingtbcmlist)
            othersmsmp.append(othersmp)
        othersmsmpall += othersmsmp
    anime.animationgen_cont(yhx, numikmsmpall, jawwidthmsmpall, objmsmpall, othersmsmpall)

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

    # base.run()
