"""
revision bassed on nextage_ros_bridge.hironx_client by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

import math
import os
import time

from hrpsys.hrpsys_config import *
import OpenHRP as OpenHRP
from distutils.version import StrictVersion

SWITCH_ON = OpenHRP.RobotHardwareService.SWITCH_ON
SWITCH_OFF = OpenHRP.RobotHardwareService.SWITCH_OFF
_MSG_ASK_ISSUEREPORT = 'Your report to ' + \
                       'https://github.com/start-jsk/rtmros_hironx/issues ' + \
                       'about the issue you are seeing is appreciated.'
_MSG_RESTART_QNX = 'You may want to restart QNX/ControllerBox afterward'

def delete_module(modname, paranoid=None):
    from sys import modules
    try:
        thismod = modules[modname]
    except KeyError:
        raise ValueError(modname)
    these_symbols = dir(thismod)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[modname]
    for mod in modules.values():
        try:
            delattr(mod, modname)
        except AttributeError:
            pass
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                try:
                    delattr(mod, symbol)
                except AttributeError:
                    pass

class HrpsysConfigurator2(HrpsysConfigurator):  ## JUST FOR TEST, REMOVE WHEN YOU MERGE
    default_frame_name = 'WAIST'

    def _get_geometry(self, method, frame_name=None):
        """!@brief
        A method only inteded for class-internal usage.

        @since: 315.12.1 or higher
        @end_type method: A Python function object.
        @param method: One of the following Python function objects defined in class HrpsysConfigurator:
            [getCurrentPose, getCurrentPosition, getCurrentReference, getCurrentRPY,
             getReferencePose, getReferencePosition, getReferenceReference, getReferenceRPY]
        @param frame_name str: set reference frame name (available from version 315.2.5)
        @rtype: {str: [float]}
        @return: Dictionary of the values for each kinematic group.
            Example (using getCurrentPosition):

                [{CHEST_JOINT0: [0.0, 0.0, 0.0]},
                 {HEAD_JOINT1: [0.0, 0.0, 0.5694999999999999]},
                 {RARM_JOINT5: [0.3255751238415409, -0.18236314012331808, 0.06762452267747099]},
                 {LARM_JOINT5: [0.3255751238415409, 0.18236314012331808, 0.06762452267747099]}]
        """
        _geometry_methods = ['getCurrentPose', 'getCurrentPosition',
                             'getCurrentReference', 'getCurrentRPY',
                             'getReferencePose', 'getReferencePosition',
                             'getReferenceReference', 'getReferenceRPY']
        if method.__name__ not in _geometry_methods:
            raise NameError("Passed method {} is not supported.".format(method))
        for kinematic_group in self.Groups:
            # The last element is usually an eef in each kinematic group,
            # although not required so.
            eef_name = kinematic_group[1][-1]
            try:
                result = method(eef_name, frame_name)
            except RuntimeError as e:
                print(str(e))
            print("{}: {}".format(eef_name, method(eef_name)))
        raise RuntimeError("Since no link name passed, ran it for all"
                           " available eefs.")

    def getCurrentPose(self, lname=None, frame_name=None):
        """!@brief
        Returns the current physical pose of the specified joint.
        cf. getReferencePose that returns commanded value.

        eg.
        \verbatim
             IN: robot_s.getCurrentPose('LARM_JOINT5')
             OUT: [-0.0017702356144599085,
              0.00019034630541264752,
              -0.9999984150158207,
              0.32556275164378523,
              0.00012155879975329215,
              0.9999999745367515,
               0.0001901314142046251,
               0.18236394191140365,
               0.9999984257434246,
               -0.00012122202968358842,
               -0.001770258707652326,
               0.07462472659364472,
               0.0,
               0.0,
               0.0,
               1.0]
        \endverbatim

        @end_type lname: str
        @param lname: Name of the link.
        @param frame_name str: set reference frame name (from 315.2.5)
        @rtype: list of float
        @return: Rotational matrix and the position of the given joint in
                 1-dimensional list, that is:
        \verbatim
                 [a11, a12, a13, x,
                  a21, a22, a23, y,
                  a31, a32, a33, z,
                   0,   0,   0,  1]
        \endverbatim
        """
        if not lname:
            self._get_geometry(self.getReferenceRPY, frame_name)
        ####
        #### for hrpsys >= 315.2.5, frame_name is supported
        ####   default_frame_name is set, call with lname:default_frame_name
        ####   frame_name is given, call with lname:frame_name
        ####   frame_name is none, call with lname
        #### for hrpsys <= 315.2.5, frame_name is not supported
        ####   frame_name is given, call with lname with warning / oerror
        ####   frame_name is none, call with lname
        if StrictVersion(self.fk_version) >= StrictVersion('315.2.5'):  ### CHANGED
            if self.default_frame_name and frame_name is None:
                frame_name = self.default_frame_name
            if frame_name and not ':' in lname:
                lname = lname + ':' + frame_name
        else:  # hrpsys < 315.2.4
            if frame_name:
                print('frame_name (' + lname + ') is not supported')  ### CHANGED
        pose = self.fk_svc.getCurrentPose(lname)
        if not pose[0]:
            raise RuntimeError("Could not find reference : " + lname)
        return pose[1].data

    def getReferencePose(self, lname=None, frame_name=None):
        """!@brief
        Returns the current commanded pose of the specified joint.
        cf. getCurrentPose that returns physical pose.

        @end_type lname: str
        @param lname: Name of the link.
        @param frame_name str: set reference frame name (from 315.2.5)
        @rtype: list of float
        @return: Rotational matrix and the position of the given joint in
                 1-dimensional list, that is:
        \verbatim
                 [a11, a12, a13, x,
                  a21, a22, a23, y,
                  a31, a32, a33, z,
                   0,   0,   0,  1]
        \endverbatim
        """
        if not lname:
            # Requires hrpsys 315.12.1 or higher.
            self._get_geometry(self.getReferenceRPY, frame_name)
        if StrictVersion(self.fk_version) >= StrictVersion('315.2.5'):  ### CHANGED
            if self.default_frame_name and frame_name is None:
                frame_name = self.default_frame_name
            if frame_name and not ':' in lname:
                lname = lname + ':' + frame_name
        else:  # hrpsys < 315.2.4
            if frame_name:
                print('frame_name (' + lname + ') is not supported')  ### CHANGED
        pose = self.fk_svc.getReferencePose(lname)
        if not pose[0]:
            raise RuntimeError("Could not find reference : " + lname)
        return pose[1].data

    def setTargetPose(self, gname, pos, rpy, tm, frame_name=None):
        """!@brief
        Move the end_type-effector to the given absolute pose.
        All d* arguments are in meter.

        @param gname str: Name of the joint group. Case-insensitive
        @param pos list of float: In meter.
        @param rpy list of float: In radian.
        @param tm float: Second to complete the command.
        @param frame_name str: Name of the frame that this particular command
                           references to.
        @return bool: False if unreachable.
        """
        print(gname, frame_name, pos, rpy, tm)
        # Same change as https://github.com/fkanehiro/hrpsys-base/pull/1113.
        # This method should be removed as part of
        # https://github.com/start-jsk/rtmros_hironx/pull/470, once
        # https://github.com/fkanehiro/hrpsys-base/pull/1063 resolves.
        if gname.upper() not in map(lambda x: x[0].upper(), self.Groups):
            print("setTargetPose failed. {} is not available in the kinematic groups. "
                  "Check available Groups (by e.g. self.Groups/robot_s.Groups). ".format(gname))
            return False
        if StrictVersion(self.seq_version) >= StrictVersion('315.2.5'):  ### CHANGED
            if self.default_frame_name and frame_name is None:
                frame_name = self.default_frame_name
            if frame_name and not ':' in gname:
                gname = gname + ':' + frame_name
        else:  # hrpsys < 315.2.4
            if frame_name and not ':' in gname:
                print('frame_name (' + gname + ') is not supported')  ### CHANGED
        result = self.seq_svc.setTargetPose(gname, pos, rpy, tm)
        if not result:
            print("setTargetPose failed. Maybe SequencePlayer failed to solve IK.\n"
                  + "Currently, returning IK result error\n"
                  + "(like the one in https://github.com/start-jsk/rtmros_hironx/issues/103)"
                  + " is not implemented. Patch is welcomed.")
        return result


class HIRONX(HrpsysConfigurator2):
    """
    @see: <a href = "https://github.com/fkanehiro/hrpsys-base/blob/master/" +
                    "python/hrpsys_config.py">HrpsysConfigurator</a>

    This class holds methods that are specific to Kawada Industries' dual-arm
    robot_s called Hiro.

    For the API doc for the derived methods, please see the parent
    class via the link above; nicely formatted api doc web page
    isn't available yet (discussed in
    https://github.com/fkanehiro/hrpsys-base/issues/268).
    """

    Groups = [['torso', ['CHEST_JOINT0']],
              ['head', ['HEAD_JOINT0', 'HEAD_JOINT1']],
              ['rarm', ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2',
                        'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']],
              ['larm', ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2',
                        'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5']]]

    OffPose = [[0],
               [0, 0],
               [25, -139, -157, 45, 0, 0],
               [-25, -139, -157, -45, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
    # With this pose the EEFs level up the tabletop surface.
    #    _InitialPose = [[0], [0, 0],
    #                    [-0.6, 0, -100, 15.2, 9.4, 3.2],
    #                    [0.6, 0, -100, -15.2, 9.4, -3.2],
    #                    [0, 0, 0, 0],
    #                    [0, 0, 0, 0]]
    InitialPose = [[0], [0, 0],
                    [-15, 0, -143, 0, 0, 0],
                    [15, 0, -143, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]
    INITPOS_TYPE_EVEN = 0
    INITPOS_TYPE_FACTORY = 1

    HandGroups = {'rhand': [2, 3, 4, 5], 'lhand': [6, 7, 8, 9]}

    RtcList = []

    # servo controller (grasper)
    sc = None
    sc_svc = None

    hrpsys_version = '0.0.0'

    _MSG_IMPEDANCE_CALL_DONE = (" call is done. This does't necessarily mean " +
                                "the function call was successful, since not " +
                                "all methods internally called return status")


    def __groups2flat(self, grouplist):
        """
        :param grouplist: [[],[],[],[]]
        :return: [float]

                 eg.
                 [[0], [10, 20],
                  [30, 40, 50, 60, 70, 80],
                  [90, 100, 110, 120, 130, 140]]
                 -->
                 [0, 10, 20, 30, 40, 50, 60, 70, 80,
                 90, 100, 110, 120, 130, 140]
        The group definition is in the very beginning of this class
        see nxtlib.hironx_client.py for details

        author: weiwei
        date: 20190417
        """

        retlist = []
        for group in grouplist:
            retlist.extend(group)
        return retlist

    def __flat2groups(self, flatlist):
        """
        :param flatlist: [float]
        :return: [[],[],[],[]]

                 eg.
                 [0, 10, 20, 30, 40, 50, 60, 70, 80,
                 90, 100, 110, 120, 130, 140]
                 -->
                 [[0], [10, 20],
                  [30, 40, 50, 60, 70, 80],
                  [90, 100, 110, 120, 130, 140]]
        The group definition is in the very beginning of this class
        see nxtlib.hironx_client.py for details

        author: weiwei
        date: 20190417
        """

        retlist = []
        index = 0
        for group in self.__groups:
            joint_num = len(group[1])
            retlist.append(flatlist[index: index + joint_num])
            index += joint_num
        return retlist

    def init(self, robotname="HiroNX(Robot)0", url=""):
        """
        Calls init from its superclass, which tries to connect RTCManager,
        looks for ModelLoader, and starts necessary RTC components. Also runs
        config, logger.
        Also internally calls setSelfGroups().

        @end_type robotname: str
        @end_type url: str
        """
        # reload for hrpsys 315.1.8
        print(self.configurator_name + "waiting ModelLoader")
        HrpsysConfigurator.waitForModelLoader(self)
        print(self.configurator_name + "start hrpsys")

        print(self.configurator_name + "finding RTCManager and RobotHardware")
        HrpsysConfigurator.waitForRTCManagerAndRoboHardware(self, robotname=robotname)
        print(self.configurator_name, "Hrpsys controller version info: ")
        if self.ms:
            print(self.configurator_name, "  ms = ", self.ms)
        if self.ms and self.ms.ref:
            print(self.configurator_name, "  ref = ", self.ms.ref)
        if self.ms and self.ms.ref and len(self.ms.ref.get_component_profiles()) > 0:
            print(self.configurator_name, "  version  = ", self.ms.ref.get_component_profiles()[0].version)
        if self.ms and self.ms.ref and len(self.ms.ref.get_component_profiles()) > 0 and StrictVersion(
                self.ms.ref.get_component_profiles()[0].version) < StrictVersion('315.2.0'):
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'hrpsys_315_1_9/hrpsys'))
            delete_module('ImpedanceControllerService_idl')
            # import AbsoluteForceSensorService_idl
            # import ImpedanceControllerService_idl

        # HrpsysConfigurator.init(self, robotname=robotname, url=url)
        self.sensors = self.getSensors(url)

        # all([rtm.findRTC(rn[0], rtm.rootnc) for rn in self.getRTCList()]) # not working somehow...
        if set([rn[0] for rn in self.getRTCList()]).issubset(set([x.name() for x in self.ms.get_components()])):
            print(self.configurator_name + "hrpsys components are already created and running")
            self.findComps(max_timeout_count=0, verbose=True)
        else:
            print(self.configurator_name + "no hrpsys components found running. creating now")
            self.createComps()

            print(self.configurator_name + "connecting hrpsys components")
            self.connectComps()

            print(self.configurator_name + "activating hrpsys components")
            self.activateComps()

            print(self.configurator_name + "setup hrpsys logger")
            self.setupLogger()

        print(self.configurator_name + "setup joint groups for hrpsys controller")
        self.setSelfGroups()

        print(self.configurator_name + '\033[32minitialized successfully\033[0m')

        # set hrpsys_version
        try:
            self.hrpsys_version = self.fk.ref.get_component_profile().version
        except:
            print(self.configurator_name + '\033[34mCould not get hrpsys_version\033[0m')
            pass
        self.setSelfGroups()
        self.hrpsys_version = self.fk.ref.get_component_profile().version

    def goOffPose(self, tm=7):
        """
        Move arms to the predefined (as member variable) pose where robot_s can
        be safely turned off.

        @end_type tm: float
        @param tm: Second to complete.
        """

        radflatoffpose = list(map(math.radians, self.__groups2flat(self.OffPose)))
        self.seq_svc.playPattern([radflatoffpose], [], [], [tm])
        self.servoOff(wait=False)

    def goInitial(self, tm=7):
        """
        Use playPattern to go Initial
        Avoiding "OfGroups" to make sure all joints can move together

        author: weiwei
        date: 20190419
        """

        radflatinitpose = list(map(math.radians, self.__groups2flat(self.InitialPose)))
        return self.seq_svc.playPattern([radflatinitpose], [], [], [tm])

    def getRTCList(self):
        """
        @see: HrpsysConfigurator.getRTCList

        @rtype [[str]]
        @rerutrn List of available components. Each element consists of a list
                 of abbreviated and full names of the component.
        """
        rtclist = [
            ['seq', "SequencePlayer"],
            ['sh', "StateHolder"],
            ['fk', "ForwardKinematics"],
            ['ic', "ImpedanceController"],
            ['el', "SoftErrorLimiter"],
            # ['co', "CollisionDetector"],
            ['sc', "ServoController"],
            ['log', "DataLogger"],
        ]
        if hasattr(self, 'rmfo'):
            self.ms.load("RemoveForceSensorLinkOffset")
            self.ms.load("AbsoluteForceSensor")
            if "RemoveForceSensorLinkOffset" in self.ms.get_factory_names():
                rtclist.append(['rmfo', "RemoveForceSensorLinkOffset"])
            elif "AbsoluteForceSensor" in self.ms.get_factory_names():
                rtclist.append(['rmfo', "AbsoluteForceSensor"])
            else:
                print("Component rmfo is not loadable.")
        return rtclist

    #
    def HandOpen(self, hand=None, effort=None):
        """
        Set the stretch between two fingers of the specified hand as
        hardcoded value (100mm), by internally calling self.setHandWidth.

        @end_type hand: str
        @param hand: Name of the hand joint group. In the default
                     setting of HIRONX, hand joint groups are defined
                     in member 'HandGroups' where 'lhand' and 'rhand'
                     are added.
        @end_type effort: int
        """
        self.setHandWidth(hand, 100, effort=effort)

    def HandClose(self, hand=None, effort=None):
        """
        Close 2-finger hand, by internally calling self.setHandWidth
        setting 0 width.

        @end_type hand: str
        @param hand: Name of the hand joint group. In the default
                     setting of HIRONX, hand joint groups are defined
                     in member 'HandGroups' where 'lhand' and 'rhand'
                     are added.
        @end_type effort: int
        """
        self.setHandWidth(hand, 0, effort=effort)

    def setHandJointAngles(self, hand, angles, tm=1):
        """
        @end_type hand: str
        @param hand: Name of the hand joint group. In the default
                     setting of HIRONX, hand joint groups are defined
                     in member 'HandGroups' where 'lhand' and 'rhand'
                     are added.
        @end_type angles: OpenHRP::ServoControllerService::dSequence.
        @param angles: List of (TODO: document). Elements are in degree.
        @param tm: Time to complete the task.
        """
        self.sc_svc.setJointAnglesOfGroup(hand, angles, tm)

    def setHandEffort(self, effort=100):
        """
        Set maximum torque for all existing hand components.
        @end_type effort: int
        """

        for i in [v for vs in self.HandGroups.values() for v in vs]:  # flatten
            self.sc_svc.setMaxTorque(i, effort)

    def setHandWidth(self, hand, width, tm=1, effort=None):
        """
        @end_type hand: str
        @param hand: Name of the hand joint group. In the default
                     setting of HIRONX, hand joint groups are defined
                     in member 'HandGroups' where 'lhand' and 'rhand'
                     are added.
        @param width: Max=100.
        @param tm: Time to complete the move.
        @end_type effort: int
        @param effort: Passed to self.setHandEffort if set. Not set by default.
        """
        if effort:
            self.setHandEffort(effort)
        if hand:
            self.setHandJointAngles(hand, self.hand_width2angles(width), tm)
        else:
            for h in self.HandGroups.keys():
                self.setHandJointAngles(h, self.hand_width2angles(width), tm)

    def moveHand(self, hand, av, tm=1):  # motion_vec av: + for open, - for close
        """
        Negate the angle value for {2, 3, 6, 7}th element in av.

        @end_type hand: str
        @param hand: Specifies hand. (TODO: List the possible values. Should be
                     listed in setHandJointAngles so just copy from its doc.)
        @end_type av: [int]
        @param av: angle of each joint of the specified arm
                  (TODO: need verified. Also what's the axis_length of the list?)
        @param tm: Time in second to complete the work.
        """
        for i in [2, 3, 6, 7]:  # do not change this line if servo is different, change HandGroups
            av[i] = -av[i]
        self.setHandJointAngles(hand, av, tm)

    def hand_width2angles(self, width):
        """
        TODO: Needs documented what this method does.

        @end_type width: float
        @return: None if the given width is invalid.
        """
        safetyMargin = 3
        l1, l2 = (41.9, 19)  # TODO: What are these variables?

        if width < 0.0 or width > (l1 + l2 - safetyMargin) * 2:
            return None

        xPos = width / 2.0 + safetyMargin
        a2Pos = xPos - l2
        a1radH = math.acos(a2Pos / l1)
        a1rad = math.pi / 2.0 - a1radH

        return a1rad, -a1rad, -a1rad, a1rad

    def setSelfGroups(self):
        """
        Set to the hrpsys.SequencePlayer the groups of links and joints that
        are statically defined as member variables (Groups) within this class.

        That said, override Groups variable if you prefer link and joint
        groups set differently.
        """
        # TODO: Accept groups variable. The name of the method sounds more
        #      natural if it accepts it.
        for item in self.Groups:
            self.seq_svc.addJointGroup(item[0], item[1])
        for k, v in self.HandGroups.iteritems():
            if self.sc_svc:
                self.sc_svc.addJointGroup(k, v)

    def isCalibDone(self):
        """
        Check whether joints have been calibrated.
        @rtype bool
        """
        if self.simulation_mode:
            return True
        else:
            rstt = self.rh_svc.getStatus()
            for item in rstt.servoState:
                if not item[0] & 1:
                    return False
        return True

    def isServoOn(self, jname='any'):
        """
        Check whether servo control has been turned on. Check is done by
        HIRONX.getActualState().servoState.
        @end_type jname: str
        @param jname: Name of a link (that can be obtained by "hiro.Groups"
                      as lists of groups).

                      Reserved values:
                      - any: This command will check all servos available.
                      - all: Same as 'any'.
        @rtype bool
        @return: If jname is specified either 'any' or 'all', return False
                 if the control of any of servos isn't available.
        """
        if self.simulation_mode:
            return True
        else:
            s_s = self.getActualState().servoState
            if jname.lower() == 'any' or jname.lower() == 'all':
                for s in s_s:
                    # print self.configurator_name, 's = ', s
                    if (s[0] & 2) == 0:
                        return False
            else:
                jid = eval('self.' + jname)
                print(self.configurator_name, s_s[jid])
                if s_s[jid][0] & 1 == 0:
                    return False
            return True
        return False

    def servoOn(self, jname='all', destroy=1, tm=3):
        """
        Turn on servo motors at joint specified.
        Joints need to be calibrated (otherwise error returns).

        *Troubleshooting*
        When this method does not seem to function as expected, try the
        following first before you report to the developer's community:

        - Manually move the arms to the safe pose where arms do not obstruct
          to anything and they can move back to the initial pose by goInitial.
          Then run the command again.
        - Make sure the emergency switch is toggled back.
        - Try running goActual() then servoOn().

        If none of the above did not solve your issue, please report with:
        - The result of this command (%ROSDISTRO% is "indigo" as of May 2017):

            Ubuntu$ rosversion hironx_ros_bridge
            Ubuntu$ dpkg -p ros-%ROSDISTRO%-hironx-ros-bridge

        @end_type jname: str
        @param jname: The value 'all' works iteratively for all servos.
        @param destroy: Not used.
        @end_type tm: float
        @param tm: Second to complete.
        @rtype: int
        @return: 1 or -1 indicating success or failure, respectively.
        """
        # check joints are calibrated
        if not self.isCalibDone():
            # waitInputConfirm('!! Calibrate Encoders with checkEncoders first !!\n\n')
            print("!! Calibrate Encoders with checkEncoders first !!\n\n")
            return -1

        # check servo state
        if self.isServoOn():
            return 1

        # check jname is acceptable
        if jname == '':
            jname = 'all'

        # self.rh_svc.power('all', SWITCH_ON)  #do not switch on before goActual

        try:
            # waitInputConfirm(
            #     '!! Robot Motion Warning (SERVO_ON) !!\n\n'
            #     'Confirm RELAY switched ON\n'
            #     'Push [OK] to switch servo ON(%s).' % (jname))
            print("!! Robot Motion Warning (SERVO_ON) !! Confirm RELAY switched ON")
        except:  # ths needs to change
            self.rh_svc.power('all', SWITCH_OFF)
            raise

        # Need to reset JointGroups.
        # See https://code.google.com/p/rtm-ros-robotics/issues/detail?id=277
        try:
            # remove jointGroups
            self.seq_svc.removeJointGroup("larm")
            self.seq_svc.removeJointGroup("rarm")
            self.seq_svc.removeJointGroup("head")
            self.seq_svc.removeJointGroup("torso")
        except:
            print(self.configurator_name,
                  'Exception during servoOn while removing JoingGroup. ' +
                  _MSG_ASK_ISSUEREPORT)
        try:
            # setup jointGroups
            self.setSelfGroups()  # restart groups
        except:
            print(self.configurator_name,
                  'Exception during servoOn while removing setSelfGroups. ' +
                  _MSG_ASK_ISSUEREPORT)

        try:
            self.goActual()  # This needs to happen before turning servo on.
            time.sleep(0.1)
            self.rh_svc.servo(jname, SWITCH_ON)
            time.sleep(2)
            # time.sleep(7)
            if not self.isServoOn(jname):
                print(self.configurator_name, 'servo on failed.')
                raise Exception
        except:
            print(self.configurator_name, 'exception occured')

        try:
            print('Move to Actual State, Just a minute.')
            self.seq_svc.playPattern([self.getActualState().angle], [], [], [tm])
        except:
            print(self.configurator_name, 'post servo on motion trouble')

        # turn on hand motors
        print('Turn on Hand Servo')
        if self.sc_svc:
            is_servoon = self.sc_svc.servoOn()
            print('Hands Servo on: ' + str(is_servoon))
            if not is_servoon:
                print('One or more hand servos failed to turn on. Make sure all hand modules are properly cabled ('
                      + _MSG_RESTART_QNX + ') and run the command again.')
                return -1
        else:
            print('hrpsys ServoController not found. Ignore this if you' +
                  ' do not intend to use hand servo (e.g. NEXTAGE Open).' +
                  ' If you do intend, then' + _MSG_RESTART_QNX +
                  ' and run the command again.')

        return 1

    def servoOff(self, jname='all', wait=True):
        """
        @end_type jname: str
        @param jname: The value 'all' works iteratively for all servos.
        @end_type wait: bool
        @rtype: int
        @return: 1 = all arm servo off. 2 = all servo on arms and hands off.
                -1 = Something wrong happened.
        """
        # do nothing for simulation
        if self.simulation_mode:
            print(self.configurator_name, 'omit servo off')
            return 0

        print('Turn off Hand Servo')
        if self.sc_svc:
            self.sc_svc.servoOff()
        # if the servos aren't on switch power off
        if not self.isServoOn(jname):
            if jname.lower() == 'all':
                self.rh_svc.power('all', SWITCH_OFF)
            return 1

        # if jname is not set properly set to all -> is this safe?
        if jname == '':
            jname = 'all'

        if wait:
            # waitInputConfirm(
            #     '!! Robot Motion Warning (Servo OFF)!!\n\n'
            #     'Push [OK] to servo OFF (%s).' % (jname))  # :
            print("!! Robot Motion Warning (Servo OFF)!!")
        try:
            self.rh_svc.servo('all', SWITCH_OFF)
            time.sleep(0.2)
            if jname == 'all':
                self.rh_svc.power('all', SWITCH_OFF)

            # turn off hand motors
            print('Turn off Hand Servo')
            if self.sc_svc:
                self.sc_svc.servoOff()

            return 2
        except:
            print(self.configurator_name, 'servo off: communication error')
            return -1

    def checkEncoders(self, jname='all', option=''):
        """
        Run the encoder checking sequence for specified joints,
        run goActual to adjust the motion_vec values, and then turn servos on.

        @end_type jname: str
        @param jname: The value 'all' works iteratively for all servos.
        @end_type option: str
        @param option: Possible values are follows (w/o double quote):\
                        "-overwrite": Overwrite calibration value.
        """
        if self.isServoOn():
            # waitInputConfirm('Servo must be off for calibration')
            print("Servo must be off for calibration")
            return
        # do not check encoders twice
        elif self.isCalibDone():
            # waitInputConfirm('System has been calibrated')
            print("System has been calibrated")
            return

        self.rh_svc.power('all', SWITCH_ON)
        msg = '!! Robot Motion Warning !!\n' \
              'Turn Relay ON.\n' \
              'Then Push [OK] to '
        if option == '-overwrite':
            msg = msg + 'calibrate(OVERWRITE MODE) '
        else:
            msg = msg + 'check '

        if jname == 'all':
            msg = msg + 'the Encoders of all.'
        else:
            msg = msg + 'the Encoder of the Joint "' + jname + '".'

        try:
            # waitInputConfirm(msg)
            print(msg)
        except:
            print("If you're connecting to the robot_s from remote, " + \
                  "make sure tunnel X (eg. -X option with ssh).")
            self.rh_svc.power('all', SWITCH_OFF)
            return 0

        is_result_hw = True
        print(self.configurator_name, 'calib-joint ' + jname + ' ' + option)
        self.rh_svc.initializeJointAngle(jname, option)
        print(self.configurator_name, 'done')
        is_result_hw = is_result_hw and self.rh_svc.power('all', SWITCH_OFF)
        self.goActual()  # This needs to happen before turning servo on.
        time.sleep(0.1)
        is_result_hw = is_result_hw and self.rh_svc.servo(jname, SWITCH_ON)
        if not is_result_hw:
            # The step described in the following msg is confirmed by the manufacturer 12/14/2015
            print(
                "Turning servos ({}) failed. This is likely because of issues happening in lower level. Please check if the Kawada's proprietary tool NextageOpenSupervisor returns without issue or not. If the issue persists, contact the manufacturer.".format(
                    jname))
        # turn on hand motors
        print('Turn on Hand Servo')
        if self.sc_svc:
            self.sc_svc.servoOn()

    def startImpedance_315_1(self, arm,
                             M_p=100.0,
                             D_p=100.0,
                             K_p=100.0,
                             M_r=100.0,
                             D_r=2000.0,
                             K_r=2000.0,
                             ref_force=[0, 0, 0],
                             force_gain=[1, 1, 1],
                             ref_moment=[0, 0, 0],
                             moment_gain=[0, 0, 0],
                             sr_gain=1.0,
                             avoid_gain=0.0,
                             reference_gain=0.0,
                             manipulability_limit=0.1):
        """
        @end_type arm: str name of artm to be controlled, this must be initialized
                   using setSelfGroups()
        @param ref_{force, moment}: Target values at the target position.
                                    Units: N, Nm, respectively.
        @param {force, moment}_gain: multipliers to the eef offset position
                                     vel_p and orientation vel_r. 3-dimensional
                                     vector (then converted internally into a
                                     diagonal matrix).
        """
        ic_sensor_name = 'rhsensor'
        ic_target_name = 'RARM_JOINT5'
        if arm == 'rarm':
            ic_sensor_name = 'rhsensor'
            ic_target_name = 'RARM_JOINT5'
        elif arm == 'larm':
            ic_sensor_name = 'lhsensor'
            ic_target_name = 'LARM_JOINT5'
        else:
            print('startImpedance: argument must be rarm or larm.')
            return

        self.ic_svc.setImpedanceControllerParam(
            OpenHRP.ImpedanceControllerService.impedanceParam(
                name=ic_sensor_name,
                base_name='CHEST_JOINT0',
                target_name=ic_target_name,
                M_p=M_p,
                D_p=D_p,
                K_p=K_p,
                M_r=M_r,
                D_r=D_r,
                K_r=K_r,
                ref_force=ref_force,
                force_gain=force_gain,
                ref_moment=ref_moment,
                moment_gain=moment_gain,
                sr_gain=sr_gain,
                avoid_gain=avoid_gain,
                reference_gain=reference_gain,
                manipulability_limit=manipulability_limit))

    def stopImpedance_315_1(self, arm):
        ic_sensor_name = 'rhsensor'
        if arm == 'rarm':
            ic_sensor_name = 'rhsensor'
        elif arm == 'larm':
            ic_sensor_name = 'lhsensor'
        else:
            print('startImpedance: argument must be rarm or larm.')
            return
        self.ic_svc.deleteImpedanceControllerAndWait(ic_sensor_name)

    def startImpedance_315_2(self, arm,
                             M_p=100.0,
                             D_p=100.0,
                             K_p=100.0,
                             M_r=100.0,
                             D_r=2000.0,
                             K_r=2000.0,
                             force_gain=[1, 1, 1],
                             moment_gain=[0, 0, 0],
                             sr_gain=1.0,
                             avoid_gain=0.0,
                             reference_gain=0.0,
                             manipulability_limit=0.1):
        """
        @end_type arm: str name of artm to be controlled, this must be initialized
                   using setSelfGroups()
        @param {force, moment}_gain: multipliers to the eef offset position
                                     vel_p and orientation vel_r. 3-dimensional
                                     vector (then converted internally into a
                                     diagonal matrix).
        """
        self.ic_svc.setImpedanceControllerParam(
            arm,
            OpenHRP.ImpedanceControllerService.impedanceParam(
                M_p=M_p,
                D_p=D_p,
                K_p=K_p,
                M_r=M_r,
                D_r=D_r,
                K_r=K_r,
                force_gain=force_gain,
                moment_gain=moment_gain,
                sr_gain=sr_gain,
                avoid_gain=avoid_gain,
                reference_gain=reference_gain,
                manipulability_limit=manipulability_limit))
        return self.ic_svc.startImpedanceController(arm)

    def startImpedance_315_3(self, arm,
                             M_p=100.0,
                             D_p=100.0,
                             K_p=100.0,
                             M_r=100.0,
                             D_r=2000.0,
                             K_r=2000.0,
                             force_gain=[1, 1, 1],
                             moment_gain=[0, 0, 0],
                             sr_gain=1.0,
                             avoid_gain=0.0,
                             reference_gain=0.0,
                             manipulability_limit=0.1):
        """
        @end_type arm: str name of artm to be controlled, this must be initialized
                   using setSelfGroups()
        @param {force, moment}_gain: multipliers to the eef offset position
                                     vel_p and orientation vel_r. 3-dimensional
                                     vector (then converted internally into a
                                     diagonal matrix).
        """
        r, p = self.ic_svc.getImpedanceControllerParam(arm)
        if not r:
            print('{}, impedance parameter not found for {}.'.format(self.configurator_name, arm))
            return False
        if M_p != None: p.M_p = M_p
        if D_p != None: p.M_p = D_p
        if K_p != None: p.M_p = K_p
        if M_r != None: p.M_r = M_r
        if D_r != None: p.M_r = D_r
        if K_r != None: p.M_r = K_r
        if force_gain != None: p.force_gain = force_gain
        if moment_gain != None: p.moment_gain = moment_gain
        if sr_gain != None: p.sr_gain = sr_gain
        if avoid_gain != None: p.avoid_gain = avoid_gain
        if reference_gain != None: p.reference_gain = reference_gain
        if manipulability_limit != None: p.manipulability_limit = manipulability_limit
        self.ic_svc.setImpedanceControllerParam(arm, p)
        return self.ic_svc.startImpedanceController(arm)

    def stopImpedance_315_2(self, arm):
        return self.ic_svc.stopImpedanceController(arm)

    def stopImpedance_315_3(self, arm):
        return self.ic_svc.stopImpedanceController(arm)

    def startImpedance(self, arm, **kwargs):
        """
        Enable the ImpedanceController RT component.
        This method internally calls startImpedance-*, hrpsys version-specific
        method.

        @requires: ImpedanceController RTC to be activated on the robot_s's
                   controller.
        @param arm: Name of the kinematic group (i.e. self.Groups[n][0]).
        @param kwargs: This varies depending on the version of hrpsys your
                       robot_s's controller runs on
                       (which you can find by "self.hrpsys_version" command).
                       For instance, if your hrpsys is 315.10.1, refer to
                       "startImpedance_315_4" method.
        @change: (NOTE: This "change" block is a duplicate with the PR in the
                 upstream https://github.com/fkanehiro/hrpsys-base/pull/1120.
                 Once it gets merged this block should be removed to avoid
                 duplicate maintenance effort.)

                 From 315.2.0 onward, following arguments are dropped and can
                 be set by self.seq_svc.setWrenches instead of this method.
                 See an example at https://github.com/fkanehiro/hrpsys-base/pull/434/files#diff-6204f002204dd9ae80f203901f155fa9R44:
                 - ref_force[fx, fy, fz] (unit: N) and
                   ref_moment[tx, ty, tz] (unit: Nm) can be set via
                   self.seq_svc.setWrenches. For example:

                   self.seq_svc.setWrenches([0, 0, 0, 0, 0, 0,
                                             fx, fy, fz, tx, ty, tz,
                                             0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0,])

                   setWrenches takes 6 values per sensor, so the robot_s in
                   the example above has 4 sensors where each line represents
                   a sensor. See this link (https://github.com/fkanehiro/hrpsys-base/pull/434/files) for a concrete example.
        """
        if StrictVersion(self.hrpsys_version) < StrictVersion('315.2.0'):
            self.startImpedance_315_1(arm, **kwargs)
        elif StrictVersion(self.hrpsys_version) < StrictVersion('315.3.0'):
            self.startImpedance_315_2(arm, **kwargs)
        else:
            self.startImpedance_315_3(arm, **kwargs)
        print('startImpedance {}'.format(self._MSG_IMPEDANCE_CALL_DONE))

    def stopImpedance(self, arm):
        if StrictVersion(self.hrpsys_version) < StrictVersion('315.2.0'):
            self.stopImpedance_315_1(arm)
        elif StrictVersion(self.hrpsys_version) < StrictVersion('315.3.0'):
            self.stopImpedance_315_2(arm)
        else:
            self.stopImpedance_315_3(arm)
        print('stopImpedance {}'.format(self._MSG_IMPEDANCE_CALL_DONE))

    def removeForceSensorOffset(self):
        self.rh_svc.removeForceSensorOffset()

    def getJointAnglesOfGroup(self, limb):
        angles = self.getJointAngles()
        offset = 0
        if len(angles) != reduce(lambda x, y: x + len(y[1]), self.Groups, 0):
            offset = 4
        angles = []
        index = 0
        for group in self.Groups:
            joint_num = len(group[1])
            angles.append(angles[index: index + joint_num])
            index += joint_num
            if group[0] in ['larm', 'rarm']:
                index += offset  ## FIX ME
        groups = self.Groups
        for i in range(len(groups)):
            if groups[i][0] == limb:
                return angles[i]
        print(self.configurator_name, 'could not find limb name ' + limb)
        print(self.configurator_name, ' in' + filter(lambda x: x[0], groups))

    def clearOfGroup(self, limb):
        """!@brief
        Clears the Sequencer's current operation for joint groups.
        @since 315.5.0
        """
        if StrictVersion(self.seq_version) < StrictVersion('315.5.0'):
            raise RuntimeError('clearOfGroup is not available with your '
                               'software version ' + self.seq_version)
        HrpsysConfigurator.clearOfGroup(self, limb)
        angles = self.getJointAnglesOfGroup(limb)
        print(self.configurator_name, 'clearOfGroup(' + limb + ') will send ' + str(
            angles) + ' to update seqplay until https://github.com/fkanehiro/hrpsys-base/pull/1141 is available')
        self.setJointAnglesOfGroup(limb, angles, 0.1, wait=True)
