import os
import database.dbaccess as db
import pandaplotutils.pandactrl as pandactrl
from wrs import manipulation as yi, manipulation as freegrip, motion as cdball
import environment.collisionmodel as cm
import environment.suitayuminotop as yumisetting
from wrs.robot_sim import yumi
from wrs.robot_sim import yumimesh, yumiball
import copy
import environment.bulletcdhelper as bch

if __name__=='__main__':

    base = pandactrl.World(camp=[2700, -2000, 2000], lookatpos=[0, 0, 0])

    env = yumisetting.Env()
    env.reparentTo(base.render)
    obscmlist = env.getstationaryobslist()
    for obscm in obscmlist:
        obscm.showcn()

    _this_dir, _ = os.path.split(__file__)
    objpath = os.path.join(_this_dir, "objects", "tubelarge.stl")
    objcm = cm.CollisionModel(objinit = objpath)

    armname = "rgt"

    hndfa = yi.YumiIntegratedFactory()
    hndfa = yi.YumiIntegratedFactory()
    rgthnd = hndfa.genHand()
    lfthnd = hndfa.genHand()
    robot = yumi.YumiRobot(rgthnd, lfthnd)
    robotball = yumiball.YumiBall()
    robotmesh = yumimesh.YumiMesh()
    pcdchecker = cdball.CollisionCheckerBall(robotball)
    bcdchecker = bch.MCMchecker(toggledebug=True)
    # robot_s.opengripper(arm_name=arm_name)
    robotnp = robotmesh.genmnp(robot)
    # robotnp.reparentTo(base.render)

    gdb = db.GraspDB(database="yumi")
    print("Loading from database...")
    fgdata = freegrip.FreegripDB(gdb, objcm.name, rgthnd.name)

    tubecmlist = []
    for x in range(250, 350, 35):
        for y in range(-200, -100, 35):
            objlttemp = copy.deepcopy(objcm)
            objlttemp.setPos(x, y, 0)
            tubecmlist.append(objlttemp)

    # tubecmlist =tubecmlist[1:]
    for tubecm in tubecmlist:
        tubecm.reparentTo(base.render)

    nptoshow  = []
    rotmat4 = tubecmlist[0].getMat()
    for i, freegriprotmat in enumerate(fgdata.freegriprotmats):
        newgriprotmat = freegriprotmat*rotmat4
        ttgsrotmat3np = base.pg.mat3ToNp(newgriprotmat.getUpper3())
        ttgscct0=rotmat4.xformPoint(fgdata.freegripcontacts[i][0])
        ttgscct1=rotmat4.xformPoint(fgdata.freegripcontacts[i][1])
        ttgsfgrcenter = (ttgscct0+ttgscct1)/2
        ttgsfgrcenternp = base.pg.v3ToNp(ttgsfgrcenter)

        yinew = hndfa.genHand()
        yinew.setMat(newgriprotmat)
        yinew.setjawwidth(50)
        jawwidth = fgdata.freegripjawwidth[i]
        isHndCollided = bcdchecker.isMeshListMeshListCollided(yinew.genRangeCmList(jawwidth, jawwidth+10), obscmlist+tubecmlist[1:])
        print(isHndCollided)
        if not isHndCollided:
            ikyumi = robot.numik(ttgsfgrcenternp, ttgsrotmat3np, armname = armname)
            if ikyumi is not None:
                robot.movearmfk(ikyumi, armname)
                isRbtCollided = pcdchecker.isRobotCollided(robot, obscmlist+tubecmlist, holdarmname=armname)
                if not isRbtCollided:
                    robotnp = robotmesh.genmnp(robot, rgbargt=[1,0,0,.3])
                    robot.opengripper(armname=armname, jawwidth = jawwidth+10)
                    # robotnp.reparentTo(base.render)
                    nptoshow.append(copy.deepcopy(robot))
                    # bcdchecker.showMeshList(tubecmlist[1:])
                    # bcdchecker.showMeshList(yinew.genRangeCmList(15))

    rbtmnp = [None, None]
    nodecounter = [0]
    def update(rbtmnp, nptoshow, nodecounter, task):
        if base.inputmgr.keyMap['space'] is True:
            if nodecounter[0] < len(nptoshow):
                if rbtmnp[0] is not None:
                    rbtmnp[0].detachNode()
                    # rbtmnp[1].detachNode()
                robot = nptoshow[nodecounter[0]]
                rbtmnp[0] =  robotmesh.genmnp(robot, rgbargt=[1,0,0,.3])
                rbtmnp[0].reparentTo(base.render)
                # rbtmnp[1] =  robotball.showfullcn(nptoshow[nodecounter[0]])
                # rbtmnp[1].reparentTo(base.render)
                nodecounter[0]+=1
                print(nodecounter[0])
            else:
                nodecounter[0] = 0
            base.inputmgr.keyMap['space'] = False
        return task.again

    taskMgr.doMethodLater(0.05, update, "update",
                          extraArgs=[rbtmnp, nptoshow, nodecounter],
                          appendTask=True)
    base.run()