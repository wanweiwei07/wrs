from wrs.motion import smoother as sm
from wrs.motion import checker as ctcb
from wrs.motion import collisioncheckerball as cdck
from wrs.robot_sim import yumi
from wrs.robot_sim import yumimesh, yumiball
from pandaplotutils import pandactrl
from wrs import manipulation as yi
import numpy as np
import environment.suitayuminotop as yumisetting
import os
import copy
import tubepuzzle as tp

base = pandactrl.World(camp=[2700, -2000, 2000], lookatpos=[0, 0, 0])

env = yumisetting.Env()
env.reparentTo(base.render)
obscmlist = env.getstationaryobslist()
for obscm in obscmlist:
    obscm.showcn()

_this_dir, _ = os.path.split(__file__)
_smalltubepath = os.path.join(_this_dir, "objects", "tubesmall.stl")
_largetubepath = os.path.join(_this_dir, "objects", "tubelarge.stl")
_tubestand = os.path.join(_this_dir, "objects", "tubestand.stl")
objst = env.loadobj(_smalltubepath)
objst.setColor(.7,.7,.7,.9)
objlt = env.loadobj(_largetubepath)
objlt.setColor(.3,.3,.3,.9)
objtsd = env.loadobj(_tubestand)

objtsd.setColor(0,.5,.7,1.9)
objtsdpos = [300,0,0]
objtsd.setPos(objtsdpos[0],objtsdpos[1],objtsdpos[2])
objtsd.reparentTo(base.render)

hndfa = yi.YumiIntegratedFactory()
rgthnd = hndfa.genHand()
lfthnd = hndfa.genHand()
robot = yumi.YumiRobot(rgthnd, lfthnd)
robot.opengripper(armname="rgt")
robot.close_gripper(armname="lft")
robotball = yumiball.YumiBall()
robotmesh = yumimesh.YumiMesh()
robotnp = robotmesh.genmnp(robot)
robotnp.reparentTo(base.render)
elearray = np.array([[0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,2,2,2,2,0,0,0],
                     [0,0,2,1,1,1,0,0,0,0],
                     [0,0,2,1,2,2,0,0,0,0],
                     [0,0,0,0,2,0,0,0,0,0]])


def getPos(i,j,objtsdpos):
    x = 300+(i-2)*19
    y = 9+(j-5)*18
    z = objtsdpos[2]+2
    return (x,y,z)
for i in range(5):
    for j in range(10):
        if elearray[i][j]==1:
            objsttemp = copy.deepcopy(objst)
            pos = getPos(i,j,objtsdpos)
            objsttemp.setPos(pos[0], pos[1], pos[2])
            objsttemp.reparentTo(base.render)
        if elearray[i][j]==2:
            objsttemp = copy.deepcopy(objlt)
            pos = getPos(i,j,objtsdpos)
            objsttemp.setPos(pos[0], pos[1], pos[2])
            objsttemp.reparentTo(base.render)
base.run()

armname = 'rgt'
cdchecker = cdck.CollisionCheckerBall(robotball)
ctcallback = ctcb.CtCallback(base, robot, cdchecker, armname=armname)
smoother = sm.Smoother()

# state = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                      [0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
#                      [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
#                      [1, 0, 0, 0, 0, 0, 0, 0, 2, 2],
#                      [1, 0, 0, 0, 0, 0, 0, 2, 0, 2]])
elearray = np.array([[0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,2,2,2,2,0,0,0],
                     [0,0,2,1,1,1,0,0,0,0],
                     [0,0,2,1,2,2,0,0,0,0],
                     [0,0,0,0,2,0,0,0,0,0]])
tpobj = tp.TubePuzzle(elearray)
path = tpobj.atarSearch()

counter = [0]
tubemnplist = [[]]
def update(path, objst, objlt, counter, tubemnplist, objtsdpos, task):
    if counter[0] < len(path):
        if len(tubemnplist[0])>0:
            for tubemnp in tubemnplist[0]:
                tubemnp.detachNode()
            tubemnplist[0] = []
        state = path[counter[0]]
        for i in range(state.nrow):
            for j in range(state.ncolumn):
                if state[i][j] == 1:
                    objsttemp = copy.deepcopy(objst)
                    pos = getPos(i,j,objtsdpos)
                    objsttemp.setPos(pos[0], pos[1], pos[2])
                    objsttemp.reparentTo(base.render)
                    tubemnplist[0].append(objsttemp)
                elif state[i][j] == 2:
                    objlttemp = copy.deepcopy(objlt)
                    pos = getPos(i,j,objtsdpos)
                    objlttemp.setPos(pos[0], pos[1], pos[2])
                    objlttemp.reparentTo(base.render)
                    tubemnplist[0].append(objlttemp)
        if base.inputmgr.keyMap['space'] is True:
            base.inputmgr.keyMap['space'] = False
            counter[0] += 1
    # else:
    #     counter[0] = 0
    return task.again
taskMgr.doMethodLater(0.05, update, "update",
                      extraArgs=[path, objst, objlt, counter, tubemnplist, objtsdpos],
                      appendTask=True)

# robot_s.goinitpose()
# starttreesamplerate = 50
# endtreesamplerate = 50
# rbtstartpos = np.array([250,-250,200])
# rbtstartrot = np.array([[1,0,0],
#                     [0,-0.92388,-0.382683],
#                     [0,0.382683,-0.92388]]).T
# rbtgoalpos = np.array([300,0,100])
# rbtgoalrot = np.dot(rm.rodrigues([0,0,1],-120),rbtstartrot)
# start = robot_s.numik(rbtstartpos, rbtstartrot, arm_name=arm_name)
# print(start)
# goal = robot_s.numik(rbtgoalpos, rbtgoalrot, arm_name=arm_name)
# print(goal)
# planner = rrtc.RRTConnect(start=start, goal=goal, ctcallback=ctcallback,
#                               starttreesamplerate=starttreesamplerate,
#                               endtreesamplerate=endtreesamplerate, expanddis=7,
#                               max_n_iter=2000, max_time=100.0)
# robot_s.movearmfk(start, arm_name)
# robotnp = robotmesh.genmnp(robot_s)
# robotnp.reparentTo(base.render)
# robot_s.movearmfk(goal, arm_name)
# robotnp = robotmesh.genmnp(robot_s)
# robotnp.reparentTo(base.render)
# robotball.showcn(robotball.genfullbcndict(robot_s))
# # base.run()
# [path, sampledpoints] = planner.planning(obscmlist+tubecmlist)
# path = smoother.pathsmoothing(path, planner, max_n_iter=100)
# print(path)
# def update(rbtmnp, motioncounter, robot_s, path, arm_name, robotmesh, robotball, task):
#     if motioncounter[0] < len(path):
#         if rbtmnp[0] is not None:
#             rbtmnp[0].detachNode()
#             rbtmnp[1].detachNode()
#         pose = path[motioncounter[0]]
#         robot_s.movearmfk(pose, arm_name)
#         rbtmnp[0] = robotmesh.genmnp(robot_s)
#         bcndict = robotball.genfullactivebcndict(robot_s)
#         rbtmnp[1] = robotball.showcn(bcndict)
#         rbtmnp[0].reparentTo(base.render)
#         motioncounter[0] += 1
#     else:
#         motioncounter[0] = 0
#     return task.again
#
# rbtmnp = [None, None]
# motioncounter = [0]
# taskMgr.doMethodLater(0.05, update, "update",
#                       extraArgs=[rbtmnp, motioncounter, robot_s, path, arm_name, robotmesh, robotball],
#                       appendTask=True)
base.run()