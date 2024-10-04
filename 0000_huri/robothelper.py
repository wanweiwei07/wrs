import numpy as np
import utiltools.robotmath as rm
from wrs.robot_sim import yumi
from wrs.robot_sim import yumimesh, yumiball
import environment.suitayuminotop as yumisetting
from wrs import manipulation as yi
from pandaplotutils import pandactrl
from wrs.motion import checker as ctcb
from wrs.motion import collisioncheckerball as cdck
from wrs.motion import rrtconnect as rrtc
from wrs.motion import smoother as sm
import utiltools.thirdparty.p3dhelper as p3dh
import environment.collisionmodel as cm
import environment.bulletcdhelper as bch
import copy
import os

class RobotHelper(object):

    def __init__(self, startworld=True, autorotate = False):
        """
        helper function to simplify everything
        :return:
        """

        self.env = yumisetting.Env()
        self.obscmlist = self.env.getstationaryobslist()

        self.rgthndfa = yi.YumiIntegratedFactory()
        self.lfthndfa = self.rgthndfa
        self.rgthnd = self.rgthndfa.genHand()
        self.lfthnd = self.lfthndfa.genHand()
        self.rbt = yumi.YumiRobot(self.rgthnd, self.lfthnd)
        self.rbtball = yumiball.YumiBall()
        self.rbtmesh = yumimesh.YumiMesh()
        self.pcdchecker = cdck.CollisionCheckerBall(self.rbtball)
        self.bcdchecker = bch.MCMchecker(toggledebug=False)
        self.ctcallback = ctcb.CtCallback(self.rbt, self.pcdchecker, armname="rgt")
        self.smoother = sm.Smoother()
        self.root = os.path.abspath(os.path.dirname(__file__))
        self.p3dh = p3dh
        self.cm = cm
        self.np = np
        self.rm = rm
        if startworld:
            self.base = self.startworld(autorotate = autorotate)

    def planmotion(self, initjnts, goaljnts, obscmlist, armname, expanddis=30):
        """

        :param initjnts:
        :param goaljnts:
        :param armname:
        :param obscmlist: objcmlist including those in the env
        :param expanddis
        :return:

        author: weiwei
        date: 20191229osaka
        """

        self.ctcallback.setarmname(armname=armname)
        starttreesamplerate = 30.0
        endtreesamplerate = 30.0
        planner = rrtc.RRTConnect(start=initjnts, goal=goaljnts, ctcallback=self.ctcallback,
                                  starttreesamplerate=starttreesamplerate,
                                  endtreesamplerate=endtreesamplerate, expanddis=expanddis,
                                  maxiter=400, maxtime=15.0)
        path, _ = planner.planning(obscmlist)
        if path is None:
            print("No path found in planning!")
            return None
        path = self.smoother.pathsmoothing(path, planner, maxiter=50)
        return path

    def planmotionhold(self, initjnts, goaljnts, objcm, relpos, relrot, obscmlist, armname, expanddis=15):
        """

        :param initjnts:
        :param goaljnts:
        :param objcm:
        :param relpos:
        :param relrot:
        :param armname:
        :param obscmlist: objcmlist including those in the env
        :return:

        author: weiwei
        date: 20191229osaka
        """

        self.ctcallback.setarmname(armname=armname)
        starttreesamplerate = 30.0
        endtreesamplerate = 30.0
        planner = rrtc.RRTConnect(start=initjnts, goal=goaljnts, ctcallback=self.ctcallback,
                                  starttreesamplerate=starttreesamplerate,
                                  endtreesamplerate=endtreesamplerate, expanddis=expanddis,
                                  maxiter=400, maxtime=15.0)
        path, _ = planner.planninghold([objcm], [[relpos, relrot]], obscmlist)
        if path is None:
            self.rbt.movearmfk(initjnts, armname)
            self.rbtmesh.genmnp(self.rbt).reparentTo(base.render)
            abpos, abrot = self.rbt.getworldpose(relpos, relrot, armname)
            objcm.set_homomat(self.rm.homobuild(abpos, abrot))
            objcm.reparentTo(base.render)
            objcm.showcn()
            for obscm in obscmlist:
                obscm.reparentTo(base.render)
                obscm.showcn()
            self.rbt.movearmfk(goaljnts, armname)
            self.rbtmesh.genmnp(self.rbt).reparentTo(base.render)
            abpos, abrot = self.rbt.getworldpose(relpos, relrot, armname)
            objcmcopy = copy.deepcopy(objcm)
            objcmcopy.set_homomat(self.rm.homobuild(abpos, abrot))
            objcmcopy.reparentTo(base.render)
            objcmcopy.showcn()
            for obscm in obscmlist:
                obscmcopy = copy.deepcopy(obscm)
                obscmcopy.reparentTo(base.render)
                obscmcopy.showcn()
            base.run()
            print("No path found in planning hold!")
        path = self.smoother.pathsmoothinghold(path, planner, maxiter=200)

        return path

    def planmotionmultiplehold(self, initjnts, goaljnts, objcmlist, objrelmatlist, obscmlist, armname):
        """

        :param initjnts:
        :param goaljnts:
        :param objcmlist:
        :param objrelmatlist: [[pos, rotmat], [pos, rotmat], ...]
        :param armname:
        :param obscmlist: objcmlist including those in the env
        :return:

        author: weiwei
        date: 20191229osaka
        """

        self.ctcallback.setarmname(armname=armname)
        starttreesamplerate = 30.0
        endtreesamplerate = 30.0
        planner = rrtc.RRTConnect(start=initjnts, goal=goaljnts, ctcallback=self.ctcallback,
                                  starttreesamplerate=starttreesamplerate,
                                  endtreesamplerate=endtreesamplerate, expanddis=10,
                                  maxiter=400, maxtime=15.0)
        path, _ = planner.planninghold(objcmlist, objrelmatlist,obscmlist)
        if path is None:
            print("No path found in planning hold!")
        path = self.smoother.pathsmoothinghold(path, planner, maxiter=100)

        return path

    def movetopose(self, homomat, armname):
        """

        :param homomat: 4x4 numpy array
        :param armname:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20191229
        """

        eepos = homomat[:3, 3]
        eerot = homomat[:3, :3]
        armjnts = self.rbt.numik(eepos, eerot, armname)
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, armname)
            return armjnts
        else:
            print("No IK solution for the given pos.")
            return None

    def movetoposrot(self, eepos, eerot, armname):
        """

        :param eepos, eerot: 1x3 nparray and 3x3 nparray
        :param armname:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20191229
        """

        armjnts = self.rbt.numik(eepos, eerot, armname)
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, armname)
            return armjnts
        else:
            print("No IK solution for the given pos, rotmat.")
            return None

    def movetoposrotmsc(self, eepos, eerot, msc, armname):
        """

        :param eepos, eerot: 1x3 nparray and 3x3 nparray
        :param msc, for ikmsc
        :param armname:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20191229
        """

        armjnts = self.rbt.numikmsc(eepos, eerot, msc, armname)
        if armjnts is not None:
            self.rbt.movearmfk(armjnts, armname)
            return armjnts
        else:
            print("No IK solution for the given pos, rotmat.")
            return None

    def genmoveforwardmotion(self, direction, distance, startarmjnts=None, obstaclecmlist=[], armname='rgt'):
        """

        :param direction:
        :param distance:
        :param startarmjnts:
        :param armname:
        :param obstaclecmlist:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20200105
        """

        if startarmjnts is None:
            startarmjnts = self.rbt.getarmjnts(armname)
        self.ctcallback.setarmname(armname=armname)
        path = self.ctcallback.getLinearPrimitive(startarmjnts, -direction, distance, [], [[]], obstaclecmlist, type="sink")
        if len(path) > 0:
            return path
        else:
            return None

    def genmovebackwardmotion(self, direction, distance, startarmjnts=None, obstaclecmlist=[], armname='rgt'):
        """

        :param direction:
        :param distance:
        :param startarmjnts:
        :param armname:
        :param obstaclecmlist:
        :return:

        author: hao, revised by weiwei
        date: 20191122, 20200105
        """

        if startarmjnts is None:
            startarmjnts = self.rbt.getarmjnts(armname)
        self.ctcallback.setarmname(armname=armname)
        path = self.ctcallback.getLinearPrimitive(startarmjnts, direction, distance, [], [[]], obstaclecmlist, type="source")
        if len(path) > 0:
            return path
        else:
            return None

    def startworld(self, autorotate = False):
        """

        :return:
        """

        # 1700, -1000, -1000, 300
        self.base = pandactrl.World(camp=[3700, -2300, 1700], lookatpos=[380, -190, 0], autocamrotate=autorotate)
        # self.base = pandactrl.World(camp=[1200, -700, 700], lookat_pos=[380, -190, 0], auto_cam_rotate=autorotate)
        # self.base = pandactrl.World(camp=[1200, -190, 1000], lookat_pos=[380, -190, 0], auto_cam_rotate=autorotate)
        self.env.reparentTo(self.base.render)
        # rbtnp = self.rbtmesh.genmnp(self.robot_s)
        # rbtnp.reparentTo(base.render)
        # base.run()
        return self.base

class RobotHelperX(RobotHelper):

    def __init__(self, startworld=True, usereal = True, autorotate = False):
        super(RobotHelperX, self).__init__(startworld=startworld, autorotate = autorotate)
        import robotconn.yumirapid.yumi_robot as yr
        import robotconn.yumirapid.yumi_state as ys
        import robotconn.rpc.phoxi.phoxi_client as pcdt

        self.yr = yr
        self.ys = ys
        if usereal:
            self.rbtx = self.yr.YuMiRobot()
            self.rbtx.calibrate_grippers()
            self.rbtx.open_grippers()
        else:
            self.rbtx = None

        # phoxi related
        self.pxc = pcdt.PhxClient(host="192.168.125.100:18300")

    def movemotionx(self, path, armname):
        armx = self.rbtx.right
        if armname is "lft":
            armx = self.rbtx.left
        statelist = []
        for armjnts in path:
            ajstate = self.ys.YuMiState(armjnts)
            statelist.append(ajstate)
        armx.movetstate_cont(statelist)

    def movetox(self, armjnts, armname):
        armx = self.rbtx.right
        if armname is "lft":
            armx = self.rbtx.left
        ajstate = self.ys.YuMiState(armjnts)
        armx.movetstate_sgl(ajstate)

    def opengripperx(self, armname):
        armx = self.rbtx.right if armname is "rgt" else self.rbtx.left
        hndfa = self.rgthndfa if armname is "rgt" else self.lfthndfa
        armx.move_gripper(hndfa.jawwidthopen/2000)

    def closegripperx(self, armname):
        armx = self.rbtx.right if armname is "rgt" else self.rbtx.left
        armx.close_gripper()

    def getarmjntsx(self, armname):
        if armname is "rgt":
            return self.rbtx.right.get_state().jnts
        elif armname is "lft":
            return self.rbtx.left.get_state().jnts
        else:
            raise ValueError("Arm name must be right or left!")

    def gethcimg(self, armname):
        if armname is "rgt":
            self.rbtx.right.write_handcamimg_ftp()
        elif armname is "lft":
            self.rbtx.left.write_handcamimg_ftp()
        else:
            raise ValueError("Arm name must be right or left!")

    def togglevac(self, toggletag, armname):
        if armname is "rgt":
            self.rbtx.right.toggle_vacuum(toggletag)
        elif armname is "lft":
            self.rbtx.left.toggle_vacuum(toggletag)

    def getpressure(self, armname):
        if armname is "rgt":
            return self.rbtx.right.get_pressure()
        elif armname is "lft":
            return self.rbtx.left.get_pressure()

    def show(self):
        """

        :return:
        """

        # base = pandactrl.World(camp=[2700, -2000, 2000], lookatp=[0, 0, 500])
        # rbtnp is the plot for the simulated robot_s
        rbtnp = self.rbtmesh.genmnp(self.rbt)
        rbtnp.setColor(1,0,.3,1)
        rbtnp.reparentTo(self.base.render)
        # rbtxnp is the plot for the real robot_s
        rbt_rbtx = copy.deepcopy(self.rbt)
        rbt_rbtx.movearmfk(self.getarmjntsx("rgt"), "rgt")
        rbt_rbtx.movearmfk(self.getarmjntsx("lft"), "lft")
        rbtxnp = self.rbtmesh.genmnp(rbt_rbtx)
        rbtxnp.reparentTo(self.base.render)
        self.base.run()

if __name__ == "__main__":
    import wrs.motion.trajectory as traj
    
    rhx = RobotHelperX(usereal=True)
    rhx.gethcimg("lft")
    # rhx.show()

    # rhx.robot_s.goinitpose()
    # print(rhx.pcdchecker.isRobotCollided(rhx.robot_s))
    rhx.movetox(rhx.rbt.initrgtjnts, armname="rgt")
    rhx.movetox(rhx.rbt.initlftjnts, armname="lft")
    # rhx.closegripperx(arm_name="rgt")
    # rhx.closegripperx(arm_name="lft")
    # rhx.opengripperx(arm_name="rgt")
    # rhx.opengripperx(arm_name="lft")
    rhx.show()
    eepos = np.array([300,-100,300])
    eerot = np.array([[0,0,1],[1,0,0],[0,1,0]]).T
    goaljnts = rhx.rbt.numik(eepos, eerot, armname="rgt")
    initjnts = rhx.getarmjntsx(armname="rgt")
    print(goaljnts)
    print(initjnts)
    path = rhx.planmotion(initjnts, goaljnts, [], armname="rgt", expanddis=30)
    print(path)
    trajobj = traj.Trajectory()
    interpolatedpath = trajobj.piecewiseinterpolation(path, sampling=5)
    rhx.movemotionx(interpolatedpath, armname="rgt")
    rhx.show()
    # rbtnp = rhx.rbtmesh.genmnp(rhx.robot_s)
    # rbtnp.reparentTo(rhx.base.render)
    # rhx.base.run()
    # goalpos = np.array([50, -400, 10])
    # goalrot = np.array([[-1,0,0], [0,1,0], [0,0,-1]])
    # armjnts = rhx.movetoposrot(eepos=goalpos, eerot=goalrot, arm_name="rgt")
    # rhx.movetox(armjnts, arm_name="rgt")
    # rhx.rbtmesh.genmnp(rhx.robot_s).reparentTo(base.render)
    # rhx.opengripperx(arm_name="rgt")
    # rhx.opengripperx(arm_name="lft")
    # rhx.moveto(rhx.robot_s.initrgtjnts, arm_name="rgt")
    # rhx.moveto(rhx.robot_s.initlftjnts, arm_name="lft")
    # print(yhx.robot_s.initrgtjnts)
    # print(yhx.robot_s.initlftjnts)

    # eepos = np.array([300,-100,300])
    # eerot = np.array([[0,0,1],[1,0,0],[0,1,0]]).T
    # arm_name = "rgt"
    # armjnts = yhx.movetoposrot(eepos=eepos, eerot=eerot, arm_name=arm_name)
    # yhx.moveto(armjnts, arm_name)

    # base = pandactrl.World(camp=[2700, -2000, 2000], lookatp=[0, 0, 500])
    # yh = YumiHelper()
    # lastarmjnts = yh.robot_s.initrgtjnts
    # for x in range(200,401,50):
    #     for y in range(-300,301,50):
    #         for z in range(150, 301, 50):
    #             armjnts = yh.movetoposrotmsc(eepos=np.array([x,y,z]), eerot = eerot, msc=lastarmjnts, arm_name="rgt")
    #             if armjnts is not None and not yh.cdchecker.isSelfCollided(yh.robot_s):
    #                 lastarmjnts = armjnts
    #                 rbtnp = yh.rbtmesh.genmnp(yh.robot_s)
    #                 rbtnp.reparentTo(base.render)
    # base.run()

    # yhx = YumiHelperX()
    # lastarmjnts = yhx.robot_s.initrgtjnts
    # for x in range(220,451,50):
    #     for y in range(-300,201,50):
    #         for z in range(150, 301, 50):
    #             armjnts = yhx.movetoposrotmsc(eepos=np.array([x,y,z]), eerot = eerot, msc=lastarmjnts, arm_name="rgt")
    #             if armjnts is not None and not yhx.cdchecker.isSelfCollided(yhx.robot_s):
    #
    #                 lastarmjnts = armjnts
    #                 yhx.moveto(armjnts, arm_name="rgt")

    # yhx.show()