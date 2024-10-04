from wrs.motion import smoother as sm
from wrs.motion import checker as ctcb
from wrs.motion import collisioncheckerball as cdck
from wrs.motion import rrtconnect as rrtc
from wrs.robot_sim import yumi
from wrs.robot_sim import yumimesh, yumiball
from pandaplotutils import pandactrl
from wrs import manipulation as yi
import numpy as np
import utiltools.robotmath as rm
import environment.suitayuminotop as yumisetting
import os
import copy
import robotconn.yumirapid.yumi_robot as yr
import robotconn.yumirapid.yumi_state as ys


if __name__ == '__main__':

    base = pandactrl.World(camp=[2700, -2000, 2000], lookatpos=[0, 0, 500])

    env = yumisetting.Env()
    env.reparentTo(base.render)
    obscmlist = env.getstationaryobslist()
    for obscm in obscmlist:
        obscm.showcn()

    _this_dir, _ = os.path.split(__file__)
    _smalltubepath = os.path.join(_this_dir, "objects", "tubesmall.stl")
    _largetubepath = os.path.join(_this_dir, "objects", "tubebig.stl")
    _tubestand = os.path.join(_this_dir, "objects", "tubestand.stl")
    objst = env.loadobj(_smalltubepath)
    objlt = env.loadobj(_largetubepath)
    objtsd = env.loadobj(_tubestand)

    objtsd.setColor(0,.5,.7,1.9)
    objtsdpos = [300,0,0]
    objtsd.setPos(objtsdpos[0],objtsdpos[1],objtsdpos[2])
    objtsd.reparentTo(base.render)

    tubecmlist = []
    elearray = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 2, 2, 2, 2, 0, 0, 0],
                         [0, 0, 2, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 2, 1, 2, 2, 0, 0, 0, 0],
                         [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]])
    def getPos(i, j, objtsdpos):
        x = 300 + (i - 2) * 19
        y = 9 + (j - 5) * 18
        z = objtsdpos[2] + 2
        return (x, y, z)
    for i in range(5):
        for j in range(10):
            if elearray[i][j] == 1:
                objsttemp = copy.deepcopy(objst)
                pos = getPos(i, j, objtsdpos)
                objsttemp.setPos(pos[0], pos[1], pos[2])
                objsttemp.reparentTo(base.render)
                tubecmlist.append(objsttemp)
            if elearray[i][j] == 2:
                objsttemp = copy.deepcopy(objlt)
                pos = getPos(i, j, objtsdpos)
                objsttemp.setPos(pos[0], pos[1], pos[2])
                objsttemp.reparentTo(base.render)
                tubecmlist.append(objsttemp)
    for tubecm in tubecmlist:
        tubecm.showcn()

    hndfa = yi.YumiIntegratedFactory()
    rgthnd = hndfa.genHand()
    lfthnd = hndfa.genHand()
    robot = yumi.YumiRobot(rgthnd, lfthnd)
    robot.opengripper(armname="rgt")
    robot.close_gripper(armname="lft")
    robotball = yumiball.YumiBall()
    robotmesh = yumimesh.YumiMesh()
    robotnp = robotmesh.genmnp(robot)
    armname = 'rgt'
    cdchecker = cdck.CollisionCheckerBall(robotball)
    ctcallback = ctcb.CtCallback(robot, cdchecker, armname=armname)
    smoother = sm.Smoother()

    robot.goinitpose()
    # lftjnts = robot_s.getarmjnts("lft")
    # rgtjnts =  robot_s.getarmjnts("rgt")

    robotreal = yr.YuMiRobot()
    lftjnts = robotreal.left.get_state().jnts
    rgtjnts = robotreal.right.get_state().jnts
    start = rgtjnts
    # print(start)
    # print(lftjnts)
    # lftpose = robotreal.left.get_pose()
    # lftp = lftpose.translation*1000
    # lftr = lftpose.rotation
    # base.pggen.plotAxis(base.render, spos=lftp, srot=lftr)
    # rgtpose = robotreal.right.get_pose()
    # rgtp = rgtpose.translation*1000
    # rgtr = rgtpose.rotation
    # base.pggen.plotAxis(base.render, spos=rgtp, srot=rgtr)
    # robot_s.movearmfk(rgtjnts, 'rgt')
    # # robot_s.movearmfk(lftjnts, 'lft')
    # robotnp = robotmesh.genmnp(robot_s)
    # robotnp.reparentTo(base.render)
    # robotball.showcn(robotball.genfullbcndict(robot_s))
    # print(cdchecker.isSelfCollided(robot_s))
    # base.run()

    starttreesamplerate = 50
    endtreesamplerate = 50
    rbtstartpos = np.array([250,-250,200])
    rbtstartrot = np.array([[1,0,0],
                        [0,-0.92388,-0.382683],
                        [0,0.382683,-0.92388]]).T
    # start = robot_s.numik(rbtstartpos, rbtstartrot, arm_name=arm_name)
    # print(start)
    rbtgoalpos = np.array([300,-200,200])
    rbtgoalrot = np.dot(rm.rodrigues([0,0,1],90),rbtstartrot)
    goal = robot.numik(rbtgoalpos, rbtgoalrot, armname=armname)
    print(goal)
    planner = rrtc.RRTConnect(start=start, goal=goal, ctcallback=ctcallback,
                                  starttreesamplerate=starttreesamplerate,
                                  endtreesamplerate=endtreesamplerate, expanddis=30,
                                  maxiter=2000, maxtime=100.0)
    robot.movearmfk(start, armname)
    robotnp = robotmesh.genmnp(robot)
    robotnp.reparentTo(base.render)
    robot.movearmfk(goal, armname)
    robotnp = robotmesh.genmnp(robot)

    robotnp.reparentTo(base.render)
    robotball.showcn(robotball.genfullbcndict(robot))
    [path, sampledpoints] = planner.planning(obscmlist+tubecmlist)
    path = smoother.pathsmoothing(path, planner, maxiter=100)
    print(path)
    for pose in path:
        robot.movearmfk(pose, armname)
        rbtmnp = robotmesh.genmnp(robot, rgbargt=[1,0,0,.3])
        rbtmnp.reparentTo(base.render)
    # base.run()
    def update(rbtmnp, motioncounter, robot, path, armname, robotmesh, robotball, task):
        if base.inputmgr.keyMap['space'] is True:
            statelist = []
            for armjnts in path:
                rgtstate = ys.YuMiState(armjnts)
                statelist.append(rgtstate)
                # lftstate = robotreal.left.get_state()
                # robotreal.goto_state_sync(lftstate, rgtstate)
                # robotreal.right.goto_state(rgtstate, wait_for_res=False)
            robotreal.right.movetstate_cont(statelist)
            return task.done
        if motioncounter[0] < len(path):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()
                rbtmnp[1].detachNode()
            pose = path[motioncounter[0]]
            robot.movearmfk(pose, armname)
            rbtmnp[0] = robotmesh.genmnp(robot)
            bcndict = robotball.genfullactivebcndict(robot)
            rbtmnp[1] = robotball.showcn(bcndict)
            rbtmnp[0].reparentTo(base.render)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again

    rbtmnp = [None, None]
    motioncounter = [0]
    taskMgr.doMethodLater(0.05, update, "update",
                          extraArgs=[rbtmnp, motioncounter, robot, path, armname, robotmesh, robotball],
                          appendTask=True)
    base.run()