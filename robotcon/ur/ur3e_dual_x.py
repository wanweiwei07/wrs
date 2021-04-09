# !/usr/bin/env python

import logging
import math
import time

import  robotconn.robotiq.rtqesgripper as rg
from robotconn.robotiq.rtqft300 import Robotiq_FT300_Sensor
from utiltools import robotmath as rm
import drivers.urx.ur_robot as urrobot
import robotconn.programbuilder as pb
import numpy as np
import threading

import socket
import struct
import os

class Ur3EDualUrx():
    """
    urx 50, right arm 51, left arm 52

    author: weiwei
    date: 20180131
    """

    def __init__(self, robotsim):
        """

        :param robotsim: for global transformation, especially in attachfirm

        author: weiwei
        date: 20191014 osaka
        """

        iprgt = '10.0.2.3'
        iplft = '10.0.2.2'
        logging.basicConfig()
        self.__lftarm = urrobot.URRobot(iplft)
        self.__lftarm.set_tcp((0, 0, 0, 0, 0, 0))
        self.__lftarm.set_payload(1.28)
        self.__rgtarm = urrobot.URRobot(iprgt)
        self.__rgtarm.set_tcp((0, 0, 0, 0, 0, 0))
        self.__rgtarm.set_payload(1.28)

        self.__hand = rg.RobotiqHE()

        self.__lftarmbase = [365, 345.0, 1330.0]
        self.__rgtarmbase = [0, -345.0, 1330.0]

        # setup server socket
        ipurx = '10.0.2.11'
        self.__urx_urmdsocket_ipad = (ipurx, 50001)
        self.__urx_urmdsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__urx_urmdsocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # self.__urx_urmdsocket.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        self.__urx_urmdsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.__urx_urmdsocket.bind(self.__urx_urmdsocket_ipad)
        self.__urx_urmdsocket.listen(5)

        self.__jointscaler = 1000000
        self.__pb = pb.ProgramBuilder()
        script_dir = os.path.dirname(__file__)
        rel_path = "urscripts_eseries/moderndriver_eseries.script"
        self.__pb.load_prog(os.path.join(script_dir, rel_path))
        # set up right arm urscript
        self.__rgtarm_urscript = self.__pb.get_program_to_run()
        self.__rgtarm_urscript = self.__rgtarm_urscript.replace("parameter_ip", self.__urx_urmdsocket_ipad[0])
        self.__rgtarm_urscript = self.__rgtarm_urscript.replace("parameter_port", str(self.__urx_urmdsocket_ipad[1]))
        self.__rgtarm_urscript = self.__rgtarm_urscript.replace("parameter_jointscaler", str(self.__jointscaler))
        # set up left arm urscript
        self.__lftarm_urscript = self.__pb.get_program_to_run()
        self.__lftarm_urscript = self.__lftarm_urscript.replace("parameter_ip", self.__urx_urmdsocket_ipad[0])
        self.__lftarm_urscript = self.__lftarm_urscript.replace("parameter_port", str(self.__urx_urmdsocket_ipad[1]))
        self.__lftarm_urscript = self.__lftarm_urscript.replace("parameter_jointscaler", str(self.__jointscaler))

        # for firm placement
        self.firmstopflag = False
        self.__robotsim = robotsim

        self.__timestep = 0.005

    @property
    def rgtarm(self):
        # read-only property
        return self.__rgtarm

    @property
    def lftarm(self):
        # read-only property
        return self.__lftarm

    def opengripper(self, speedpercentage=5, forcepercentage=10, distance=100, armname="rgt"):
        """
        open the rtq85 hand on the arm specified by armname

        :param armname:
        :return:

        author: weiwei
        date: 20180220
        """

        targetarm = self.__rgtarm
        if armname == 'lft':
            targetarm = self.__lftarm
        pr = self.__hand.get_program_to_run(mode="open", speedpercentage=speedpercentage,
                                            forcepercentage=forcepercentage, distance=distance)
        targetarm.send_program(pr)
        time.sleep(1)
        while targetarm.is_program_running():
            continue

    def closegripper(self, speedpercentage=100, forcepercentage=100, armname='rgt'):
        """
        close the rtq85 hand on the arm specified by armname

        :param armname:
        :return:

        author: weiwei
        date: 20180220
        """

        targetarm = self.__rgtarm
        if armname == 'lft':
            targetarm = self.__lftarm
        pr = self.__hand.get_program_to_run(mode="close", speedpercentage=speedpercentage,
                                            forcepercentage=forcepercentage)
        targetarm.send_program(pr)
        time.sleep(1)
        while targetarm.is_program_running():
            continue

    def movejntsin360(self):
        """
        the function move all joints back to -360,360
        due to multi R problem, the last joint could be out of 360
        this function moves the last joint back

        :return:

        author: weiwei
        date: 20180202
        """

        rgtarmjnts = self.getjnts('rgt')
        lftarmjnts = self.getjnts('lft')
        rgtarmjnts = rm.cvtRngPM360(rgtarmjnts)
        lftarmjnts = rm.cvtRngPM360(lftarmjnts)
        self.movejntssgl(rgtarmjnts, armname='rgt')
        self.movejntssgl(lftarmjnts, armname='lft')

    def movejntssgl(self, joints, armname='rgt', wait=True):
        """

        :param joints: a 1-by-6 list in degree
        :param armname:
        :return:

        author: weiwei
        date: 20170411
        """

        targetarm = self.__rgtarm
        if armname == 'lft':
            targetarm = self.__lftarm

        jointsrad = [math.radians(angdeg) for angdeg in joints]
        targetarm.movej(jointsrad, acc = 1, vel = 1, wait = wait)

    def movejntsall(self, joints, wait=True):
        """
        move all joints of the ur5 dual-arm robot_s
        NOTE that the two arms are moved sequentially
        use wait=False for simultaneous motion

        :param joints:  a 1-by-12 vector in degree, 6 for right, 6 for left
        :return: bool

        author: weiwei
        date: 20170411
        """

        jointsrad = [math.radians(angdeg) for angdeg in joints[0:6]]
        self.__rgtarm.movej(jointsrad, wait = wait)
        jointsrad = [math.radians(angdeg) for angdeg in joints[6:12]]
        self.__lftarm.movej(jointsrad, wait = wait)

    def movetposesgl_cont(self, tposelist, armname='rgt', acc = 1, vel = .1, radius = 0.1, wait = True):
        """
        move robot_s continuously by inputing a list of tcp poses

        :param tposelist:
        :param armname:
        :param acc:
        :param vel:
        :param radius:
        :return:

        author: weiwei
        date: 20180420
        """

        targetarm = self.__rgtarm
        if armname == 'lft':
            targetarm = self.__lftarm

        targetarm.movels(tposelist, acc=acc, vel=vel, radius=radius, wait=wait, threshold=None)

    def movejntssgl_cont(self, jointspath, armname='rgt', timepathstep = 1.0, inpfunc = "cubic", wait = True):
        """
        move robot_s continuously using servoj and urscript

        :param jointspath: a list of joint angles as motion path
        :param armname:
        :param timepathstep: time to move between adjacent joints, timepathstep = expandis/speed, speed = degree/second
                             by default, the value is 1.0 and the speed is expandis/second
        :param inpfunc: call cubic by default, candidate values: cubic or quintic
        :return:

        author: weiwei
        date: 20180518
        """

        def cubic(t, timestamp, q0array, v0array, q1array, v1array):
            a0 = q0array
            a1 = v0array
            a2 = (-3*(q0array-q1array)-(2*v0array+v1array)*timestamp)/(timestamp**2)
            a3 = (2*(q0array-q1array)+(v0array+v1array)*timestamp)/(timestamp**3)
            qt = a0+a1*t+a2*(t**2)+a3*(t**3)
            vt = a1+2*a2*t+3*a3*(t**2)
            return qt.tolist(), vt.tolist()

        def quintic(t, timestamp, q0array, v0array,
                  q1array, v1array, a0array=np.array([0.0]*6), a1array=np.array([0.0]*6)):
            a0 = q0array
            a1 = v0array
            a2 = a0array/2.0
            a3 = (20*(q1array-q0array)-(8*v1array+12*v0array)*timestamp-
                  (3*a1array-a0array)*(timestamp**2))/(2*(timestamp**3))
            a4 = (30*(q0array-q1array)+(14*v1array+16*v0array)*timestamp+
                  (3*a1array-2*a0array)*(timestamp**2))/(2*(timestamp**4))
            a5 = (12*(q1array-q0array)-6*(v1array+v0array)*timestamp-
                  (a1array-a0array)*(timestamp**2))/(2*(timestamp**5))
            qt = a0+a1*t+a2*(t**2)+a3*(t**3)+a4*(t**4)+a5*(t**5)
            vt = a1+2*a2*t+3*a3*(t**2)+4*a4*(t**3)+5*a5*(t**4)
            return qt.tolist(), vt.tolist()

        if inpfunc != "cubic" and inpfunc != "quintic":
            raise ValueError("Interpolation functions must be cubic or quintic")
        inpfunccallback = cubic
        if inpfunc == "quintic":
            inpfunccallback = quintic

        timesstamplist = []
        speedsradlist = []
        jointsradlist = []
        for id, joints in enumerate(jointspath):
            jointsrad = [math.radians(angdeg) for angdeg in joints[0:6]]
            # print jointsrad
            jointsradlist.append(jointsrad)
            if id == 0:
                timesstamplist.append([0.0]*6)
            else:
                timesstamplist.append([timepathstep]*6)
            if id == 0 or id == len(jointspath)-1:
                speedsradlist.append([0.0]*6)
            else:
                thisjointsrad = jointsrad
                prejointsrad = [math.radians(angdeg) for angdeg in jointspath[id-1][0:6]]
                nxtjointsrad = [math.radians(angdeg) for angdeg in jointspath[id+1][0:6]]
                presarray = (np.array(thisjointsrad)-np.array(prejointsrad))/timepathstep
                nxtsarray = (np.array(nxtjointsrad)-np.array(thisjointsrad))/timepathstep
                # set to 0 if signs are different
                selectid = np.where((np.sign(presarray)+np.sign(nxtsarray))==0)
                sarray = (presarray+nxtsarray)/2.0
                sarray[selectid]=0.0
                # print presarray
                # print nxtsarray
                # print sarray
                speedsradlist.append(sarray.tolist())
        t = 0
        timestep = self.__timestep
        jointsradlisttimestep = []
        speedsradlisttimestep = []
        for idlist, timesstamp in enumerate(timesstamplist):
            if idlist == 0:
                continue
            timesstampnp = np.array(timesstamp)
            jointsradprenp = np.array(jointsradlist[idlist-1])
            speedsradprenp = np.array(speedsradlist[idlist-1])
            jointsradnp = np.array(jointsradlist[idlist])
            speedsradnp = np.array(speedsradlist[idlist])
            # reduce timestep in the last step to avoid overfitting
            if idlist == len(timesstamplist)-1:
                while t <= timesstampnp.max():
                    jsrad, vsrad = inpfunccallback(t, timesstampnp,
                                                   jointsradprenp, speedsradprenp,
                                                   jointsradnp, speedsradnp)
                    jointsradlisttimestep.append(jsrad)
                    speedsradlisttimestep.append(vsrad)
                    t = t+timestep/3
            else:
                while t <= timesstampnp.max():
                    jsrad, vsrad = inpfunccallback(t, timesstampnp,
                                                   jointsradprenp, speedsradprenp,
                                                   jointsradnp, speedsradnp)
                    jointsradlisttimestep.append(jsrad)
                    speedsradlisttimestep.append(vsrad)
                    t = t+timestep
                t = 0

        ## for debug (show the curves in pyplot)
        # import matplotlib.pyplot as plt
        # for id in range(6):
        #     plt.subplot(121)
        #     q0list008 = [joint[id] for joint in jointsradlist008]
        #     plt.plot(np.linspace(0,0.008*len(jointsradlist008),len(jointsradlist008)),q0list008)
        #     plt.subplot(122)
        #     v0list008 = [speed[id] for speed in speedsradlist008]
        #     plt.plot(np.linspace(0,0.008*len(speedsradlist008),len(speedsradlist008)),v0list008)
        # plt.show()

        # for jointsrad in jointsradlist008:
        #     print jointsrad
        # print len(jointsradlist008)

        arm = self.__rgtarm
        arm_urscript = self.__rgtarm_urscript
        if armname == 'lft':
            arm = self.__lftarm
            arm_urscript = self.__lftarm_urscript
        arm.send_program(arm_urscript)
        # accept arm socket
        urmdsocket, urmdsocket_addr = self.__urx_urmdsocket.accept()
        print("Connected by ", urmdsocket_addr)

        keepalive = 1
        buf = bytes()
        for id, jointsrad in enumerate(jointsradlisttimestep):
            if id == len(jointsradlisttimestep)-1:
                keepalive = 0
            jointsradint = [int(jointrad*self.__jointscaler) for jointrad in jointsrad]
            buf += struct.pack('!iiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                                    jointsradint[3], jointsradint[4], jointsradint[5], keepalive)
        urmdsocket.send(buf)
        if wait:
            time.sleep(.5)
            while arm.is_program_running():
                pass
        urmdsocket.close()

    def movejntssgl_cont2(self, jointspath, armname='rgt', timepathstep = 1.0, inpfunc = "cubic", wait=True):
        """
        move robot_s continuously using servoj and urscript
        movejntssgl_cont2 aims at smooth slow down motion

        :param jointspath: a list of joint angles as a motion path
        :param armname:
        :param timepathstep: time to move between adjacent joints, timepathstep = expandis/speed, speed = degree/second
                             by default, the value is 1.0 and the speed is expandis/second
        :param inpfunc: call cubic by default, candidate values: cubic or quintic
        :return:

        author: weiwei
        date: 20180606
        """

        def cubic(t, timestamp, q0array, v0array, q1array, v1array):
            a0 = q0array
            a1 = v0array
            a2 = (-3*(q0array-q1array)-(2*v0array+v1array)*timestamp)/(timestamp**2)
            a3 = (2*(q0array-q1array)+(v0array+v1array)*timestamp)/(timestamp**3)
            qt = a0+a1*t+a2*(t**2)+a3*(t**3)
            vt = a1+2*a2*t+3*a3*(t**2)
            return qt.tolist(), vt.tolist()

        def quintic(t, timestamp, q0array, v0array,
                  q1array, v1array, a0array=np.array([0.0]*6), a1array=np.array([0.0]*6)):
            a0 = q0array
            a1 = v0array
            a2 = a0array/2.0
            a3 = (20*(q1array-q0array)-(8*v1array+12*v0array)*timestamp-
                  (3*a1array-a0array)*(timestamp**2))/(2*(timestamp**3))
            a4 = (30*(q0array-q1array)+(14*v1array+16*v0array)*timestamp+
                  (3*a1array-2*a0array)*(timestamp**2))/(2*(timestamp**4))
            a5 = (12*(q1array-q0array)-6*(v1array+v0array)*timestamp-
                  (a1array-a0array)*(timestamp**2))/(2*(timestamp**5))
            qt = a0+a1*t+a2*(t**2)+a3*(t**3)+a4*(t**4)+a5*(t**5)
            vt = a1+2*a2*t+3*a3*(t**2)+4*a4*(t**3)+5*a5*(t**4)
            return qt.tolist(), vt.tolist()

        if inpfunc != "cubic" and inpfunc != "quintic":
            raise ValueError("Interpolation functions must be cubic or quintic")
        inpfunccallback = cubic
        if inpfunc == "quintic":
            inpfunccallback = quintic

        timepathsteplastratio = .25
        timesstamplist = []
        speedsradlist = []
        jointsradlist = []
        for id, joints in enumerate(jointspath):
            jointsrad = [math.radians(angdeg) for angdeg in joints[0:6]]
            # print jointsrad
            jointsradlist.append(jointsrad)
            if id == 0:
                timesstamplist.append([0.0]*6)
            else:
                timesstamplist.append([timepathstep]*6)
            if id == 0 or id == len(jointspath)-1:
                speedsradlist.append([0.0]*6)
            else:
                thisjointsrad = jointsrad
                prejointsrad = [math.radians(angdeg) for angdeg in jointspath[id-1][0:6]]
                nxtjointsrad = [math.radians(angdeg) for angdeg in jointspath[id+1][0:6]]
                presarray = (np.array(thisjointsrad)-np.array(prejointsrad))/timepathstep
                nxtsarray = (np.array(nxtjointsrad)-np.array(thisjointsrad))/timepathstep
                if id+1 == len(jointspath)-1:
                    nxtsarray = nxtsarray/timepathsteplastratio
                # set to 0 if signs are different
                selectid = np.where((np.sign(presarray)+np.sign(nxtsarray))==0)
                sarray = (presarray+nxtsarray)/2.0
                sarray[selectid]=0.0
                # print presarray
                # print nxtsarray
                # print sarray
                speedsradlist.append(sarray.tolist())
        t = 0
        timestep = self.__timestep
        jointsradlisttimestep = []
        speedsradlisttimestep = []
        for idlist, timesstamp in enumerate(timesstamplist):
            if idlist == 0:
                continue
            timesstampnp = np.array(timesstamp)
            jointsradprenp = np.array(jointsradlist[idlist-1])
            speedsradprenp = np.array(speedsradlist[idlist-1])
            jointsradnp = np.array(jointsradlist[idlist])
            speedsradnp = np.array(speedsradlist[idlist])
            while t <= timesstampnp.max():
                jsrad, vsrad = inpfunccallback(t, timesstampnp,
                                               jointsradprenp, speedsradprenp,
                                               jointsradnp, speedsradnp)
                jointsradlisttimestep.append(jsrad)
                speedsradlisttimestep.append(vsrad)
                t = t+timestep
            t = 0

        ## for debug (show the curves in pyplot)
        # import matplotlib.pyplot as plt
        # for id in range(6):
        #     plt.subplot(121)
        #     q0list008 = [joint[id] for joint in jointsradlist008]
        #     plt.plot(np.linspace(0,0.008*len(jointsradlist008),len(jointsradlist008)),q0list008)
        #     plt.subplot(122)
        #     v0list008 = [speed[id] for speed in speedsradlist008]
        #     plt.plot(np.linspace(0,0.008*len(speedsradlist008),len(speedsradlist008)),v0list008)
        # plt.show()

        # for jointsrad in jointsradlist008:
        #     print jointsrad
        # print len(jointsradlist008)

        arm = self.__rgtarm
        arm_urscript = self.__rgtarm_urscript
        if armname == 'lft':
            arm = self.__lftarm
            arm_urscript = self.__lftarm_urscript
        arm.send_program(arm_urscript)
        # accept arm socket
        urmdsocket, urmdsocket_addr = self.__urx_urmdsocket.accept()
        print("Connected by ", urmdsocket_addr)

        keepalive = 1
        buf = bytes()
        for id, jointsrad in enumerate(jointsradlisttimestep):
            if id == len(jointsradlisttimestep)-1:
                keepalive = 0
            jointsradint = [int(jointrad*self.__jointscaler) for jointrad in jointsrad]
            buf += struct.pack('!iiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                                    jointsradint[3], jointsradint[4], jointsradint[5], keepalive)
        urmdsocket.send(buf)
        if wait:
            time.sleep(.5)
            while arm.is_program_running():
                pass

        urmdsocket.close()

    def attachfirm(self, direction = np.array([0,0,-1]), steplength = 1, forcethreshold = 10, armname = 'rgt'):
        """
        place the object firmly on a table considering forcefeedback

        :param armname:
        :return:
        """

        if self.__robotsim is None:
            print("Robotsim must be passed to the construction function to call this method.")
            raise ValueError()

        originaljnts = self.__robotsim.getarmjnts(armname=armname)
        currentjnts = self.getjnts(armname=armname)
        self.__robotsim.movearmfk(currentjnts, armname=armname)
        eepos, eerot = self.__robotsim.getee(armname=armname)

        self.zerotcpforce()
        while True:
            ftdata = self.getinhandtcpforce()
            attachforce = ftdata[0] * eerot[:3, 0] + ftdata[1] * eerot[:3, 1] + ftdata[2] * eerot[:3, 2]
            force = np.linalg.norm(np.dot(attachforce, -direction))
            print(force)
            if force > forcethreshold:
                self.__robotsim.movearmfk(originaljnts)
                return
            eepos, eerot = self.__robotsim.getee()
            currentjnts = self.__robotsim.getarmjnts()
            eepos = eepos + direction * steplength
            newjnts = self.__robotsim.numikmsc(eepos, eerot, currentjnts)
            print(currentjnts)
            print(newjnts)
            self.__robotsim.movearmfk(newjnts)
            self.movejntssgl(newjnts, wait=False)
            time.sleep(.001)

    def zerotcpforce(self, armname = 'rgt'):
        prog = "zero_ftsensor()"
        arm = self.__rgtarm
        if armname == 'lft':
            arm = self.__lftarm
        arm.send_program(prog)

    def getinhandtcpforce(self, armname = 'rgt'):
        arm = self.__rgtarm
        if armname == 'lft':
            arm = self.__lftarm
        return arm.get_tcp_force()

    def getworldtcpforce(self, armname = 'rgt'):
        if self.__robotsim is None:
            print("Robotsim must be passed to the construction function to call this method.")
            raise ValueError()

        originaljnts = self.__robotsim.getarmjnts(armname=armname)
        currentjnts = self.getjnts(armname=armname)
        self.__robotsim.movearmfk(currentjnts, armname=armname)
        eepos, eerot = self.__robotsim.getee(armname=armname)
        ftdata = self.getinhandtcpforce(armname=armname)
        fworld = [ftdata[0]*eerot[:3, 0], ftdata[1]*eerot[:3, 1], ftdata[2]*eerot[:3, 2]]
        tworld = [ftdata[3]*eerot[:3, 0], ftdata[4]*eerot[:3, 1], ftdata[5]*eerot[:3, 2]]
        self.__robotsim.movearmfk(originaljnts, armname=armname)
        return fworld + tworld

    def getjnts(self, armname = 'rgt'):
        """
        get the joint angles of the specified arm

        :param armname:
        :return:

        author: ochi
        date: 20180410
        """

        targetarm = self.__rgtarm
        if armname == 'lft':
            targetarm = self.__lftarm
        armjnts_radian = targetarm.getj()
        armjnts_degree = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, ajr in enumerate(armjnts_radian):
            armjnts_degree[i] = math.degrees(ajr)

        return armjnts_degree

if __name__ == '__main__':
    import robotsim.ur3edual.ur3edual as u3ed
    import pandaplotutils.pandactrl as pc
    import manipulation.grip.robotiqhe.robotiqhe as rtqhe
    import robotsim.ur3edual.ur3edual as robot

    base = pc.World(camp = [3000,0,3000], lookatp = [0,0,700])

    ur3edualrobot = u3ed.Ur3EDualRobot()
    ur3edualrobot.goinitpose()
    ur3eu = Ur3EDualUrx(ur3edualrobot)
    #
    # hndfa = rtqhe.RobotiqHEFactory()
    # rgthnd = hndfa.genHand()
    # lfthnd = hndfa.genHand()
    #
    # rbt = robot_s.Ur3EDualRobot(rgthnd, lfthnd)
    # rbt.goinitpose()
    # ur3eu.attachfirm(rbt, upthreshold=10, armname='lft')
    ur3eu.opengripper(armname="lft",forcepercentage=0,distance=23)
    ur3eu.opengripper(armname="lft",forcepercentage=0 , distance=80)
    # ur3eu.closegripper(armname="lft")
    # initpose = ur3dualrobot.initjnts
    # initrgt = initpose[3:9]
    # initlft = initpose[9:15]
    # ur3u.movejntssgl(initrgt, armname='rgt')
    # ur3u.movejntssgl(initlft, armname='lft')

    # goalrgt = copy.deepcopy(initrgt)
    # goalrgt[0] = goalrgt[0]-10.0
    # goalrgt1 = copy.deepcopy(initrgt)zr
    # goalrgt1[0] = goalrgt1[0]-5.0
    # goallft = copy.deepcopy(initlft)
    # goallft[0] = goallft[0]+10.0

    # ur3u.movejntssgl_cont([initrgt, goalrgt, goalrgt1], armname='rgt')
    #
    # postcp_robot, rottcp_robot =  ur3dualrobot.gettcp_robot(armname='rgt')
    # print math3d.Transform(rottcp_robot, postcp_robot).get_pose_vector()
    # print "getl ", ur3u.rgtarm.getl()
    #
    # postcp_robot, rottcp_robot =  ur3dualrobot.gettcp_robot(armname='lft')
    # print math3d.Transform(rottcp_robot, postcp_robot).get_pose_vector()
    # print "getl ", ur3u.lftarm.getl()
    #
    # # tcpsimrobot =  ur5dualrobot.lftarm[-1]['linkpos']
    # # print tcprobot
    # # print tcpsimrobot
    # u3dmgen = u3dm.Ur3DualMesh(rgthand, lfthand)
    # ur3dualmnp = u3dmgen.genmnp(ur3dualrobot, togglejntscoord=True)
    # ur3dualmnp.reparentTo(base.render)
    # # armname = 'rgt'
    # # armname = 'lft'
    # # ur3u.movejntsall(ur3dualrobot.initjnts)
    # # ur3u.movejntsin360()
    # # print ur3u.getjnts('rgt')
    # # print ur3u.getjnts('lft')

    base.run()