import logging
import math
import time
from robotconn.robotiq.rtqgripper import Robotiq_Two_Finger_Gripper
from utiltools import robotmath as rm
import urx.urrobot as urrobot
import robotconn.programbuilder as pb
import numpy as np

import socket
import struct
import os

class Ur3EUrx():
    """

    author: weiwei
    date: 20180131
    """

    def __init__(self, robotsim = None):
        """

        :param robotsim: for global transformation, especially in attachfirm

        author: weiwei
        date: 20191014 osaka
        """

        ip = '10.2.0.50'

        logging.basicConfig()
        self.__arm = urrobot.URRobot(ip, use_rt=True)
        self.__arm.set_flange()
        self.__arm.set_payload(1.28)

        self.__hand = Robotiq_Two_Finger_Gripper(type=50)

        self.__armbase = [0, 235.00, 965.00]
        self.__sqrt2o2 = math.sqrt(2.0)/2.0

        # setup server socket
        ipurx = '10.2.0.100'
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
        self.__arm_urscript = self.__pb.get_program_to_run()
        self.__arm_urscript = self.__arm_urscript.replace("parameter_ip", self.__urx_urmdsocket_ipad[0])
        self.__arm_urscript = self.__arm_urscript.replace("parameter_port", str(self.__urx_urmdsocket_ipad[1]))
        self.__arm_urscript = self.__arm_urscript.replace("parameter_jointscaler", str(self.__jointscaler))

        # for firm placement
        self.firmstopflag = False
        self.__robotsim = robotsim

        self.__timestep = 0.005

    @property
    def arm(self):
        # read-only property
        return self.__arm

    def opengripper(self, speedpercentange = 70, forcepercentage = 50, fingerdistance = 50):
        """
        open the rtq85 hand on the arm specified by arm_name

        :param arm_name:
        :return:

        author: weiwei
        date: 20180220
        """

        targetarm = self.__arm
        self.__hand.open_gripper(speedpercentange, forcepercentage, fingerdistance)
        targetarm.send_program(self.__hand.get_program_to_run())

    def closegripper(self, speedpercentange = 80, forcepercentage = 50):
        """
        close the rtq85 hand on the arm specified by arm_name

        :param arm_name:
        :return:

        author: weiwei
        date: 20180220
        """

        targetarm = self.__arm
        self.__hand.close_gripper(speedpercentange, forcepercentage)
        targetarm.send_program(self.__hand.get_program_to_run())

    def movejntsin360(self):
        """
        the function move all joints back to -360,360
        due to multi R problem, the last joint could be out of 360
        this function moves the last joint back

        :return:

        author: weiwei
        date: 20180202
        """

        armjnts = self.getjnts()
        armjnts = rm.cvtRngPM360(armjnts)
        self.movejntssgl(armjnts)

    def movejnts(self, joints, wait=True):
        """

        :param joints: a 1-by-6 list in degree
        :param arm_name:
        :return:

        author: weiwei
        date: 20170411
        """

        targetarm = self.__arm

        jointsrad = [math.radians(angdeg) for angdeg in joints]
        targetarm.movej(jointsrad, acc = 1, vel = 1, wait = wait)

    def movejnts_cont(self, jointspath, timepathstep = 1.0, inpfunc = "cubic"):
        """
        move robot_s continuously using servoj and urscript

        :param jointspath: a list of joint angles as motion path
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

        arm = self.__arm
        arm_urscript = self.__arm_urscript
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

        urmdsocket.close()

    def movejnts_cont2(self, jointspath, timepathstep = 1.0, inpfunc = "cubic"):
        """
        move robot_s continuously using servoj and urscript
        movejntssgl_cont2 aims at smooth slow down motion

        :param jointspath: a list of joint angles as a motion path
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

        arm = self.__arm
        arm_urscript = self.__arm_urscript
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

        urmdsocket.close()

    def attachfirm(self, direction = np.array([0,0,-1]), steplength = .3, forcethreshold = 5):
        """
        place the object firmly on a table considering forcefeedback

        :param arm_name:
        :return:
        """

        if self.__robotsim is None:
            print("Robotsim must be passed to the construction function to call this method.")
            raise ValueError()

        originaljnts = self.__robotsim.getarmjnts()
        currentjnts = self.getjnts()
        self.__robotsim.movearmfk(currentjnts)
        eepos, eerot = self.__robotsim.getee()

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
            self.movejnts(newjnts, wait=False)
            time.sleep(.001)

    def zerotcpforce(self):
        prog = "zero_ftsensor()"
        self.__arm.send_program(prog)

    def getinhandtcpforce(self):
        return self.__arm.get_tcp_force()

    def getworldtcpforce(self):

        if self.__robotsim is None:
            print("Robotsim must be passed to the construction function to call this method.")
            raise ValueError()

        originaljnts = self.__robotsim.getarmjnts()
        currentjnts = self.getjnts()
        self.__robotsim.movearmfk(currentjnts)
        eepos, eerot = self.__robotsim.getee()
        ftdata = self.getinhandtcpforce()
        fworld = [ftdata[0]*eerot[:3, 0], ftdata[1]*eerot[:3, 1], ftdata[2]*eerot[:3, 2]]
        tworld = [ftdata[3]*eerot[:3, 0], ftdata[4]*eerot[:3, 1], ftdata[5]*eerot[:3, 2]]
        self.__robotsim.movearmfk(originaljnts)
        return fworld + tworld

    def getjnts(self):
        """
        get the joint angles of the specified arm

        :param arm_name:
        :return:

        author: ochi
        date: 20180410
        """

        targetarm = self.__arm
        armjnts_radian = targetarm.getj()
        armjnts_degree = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, ajr in enumerate(armjnts_radian):
            armjnts_degree[i] = math.degrees(ajr)

        return armjnts_degree

if __name__ == '__main__':
    from wrs import robot_sim as u3e, robot_sim as u3emesh, manipulation as rtqhe
    import pandaplotutils.pandactrl as pc

    base = pc.World(camp = [3000,0,3000], lookatpos= [0, 0, 700])

    rfa = rtqhe.HandFactory()
    hnd = rfa.genHand()
    ur3erobot = u3e.Ur3ERobot(hnd)
    ur3erobot.goinitpose()
    ur3emg = u3emesh.Ur3EMesh()
    ur3emnp = ur3emg.genmnp(ur3erobot)
    ur3emnp.reparentTo(base.render)

    ur3e = Ur3EUrx(ur3erobot)
    jnts = ur3e.getjnts()
    # joints[0]+=10
    # ur3e.movejnts(joints)
    # base.run()
    ur3erobot.movearmfk(jnts)
    ur3emg = u3emesh.Ur3EMesh()
    ur3emnp = ur3emg.genmnp(ur3erobot)
    ur3emnp.reparentTo(base.render)
    # ur3e.closegripper()
    ur3e.attachfirm()
    # ur3e.zerotcpforce()
    # def update(ur3e, task):
    #     print(ur3e.gettcpforce())
    #     return task.cont
    # taskMgr.doMethodLater(0.05, update, "update",
    #                   extraArgs=[ur3e],
    #                   appendTask=True)
    base.run()

    hndfa = rtqhe.HandFactory()
    hnd = hndfa.genHand()
    ur3e.placefirm(ur3erobot, upthreshold=10)
    ur3e.opengripper()

    base.run()