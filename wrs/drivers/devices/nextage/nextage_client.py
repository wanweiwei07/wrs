"""
revision bassed on nextage_ros_bridge.hironx_client by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

from .hironx_client import HIRONX
from .hands_05 import Hands05
from .iros13_hands import Iros13Hands

class NextageClient(HIRONX, object):

    # The 2nd arg 'object' is passed to work around the issue raised because
    # HrpsysConfigurator is "old-style" python class.
    # See http://stackoverflow.com/a/18392639/577001
    """
    This class holds methods that are specific to Kawada Industries' dual-arm
    robot_s called Nextage Open.
    """

    """ Overriding a variable in the superclass to set the arms at higher
    positions."""
    OffPose = [[0], [0, 0],
               [25, -140, -150, 45, 0, 0],
               [-25, -140, -150, -45, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]

    # Default digital input groups defined by manufacturer, Kawada, as of
    # July 2014. This may change per the robot_s in the future and in then
    # need modified. See also readDinGroup method.
    _DI_PORTS_L = [25, 21, 22, 23, 24]
    _DI_PORTS_R = [20, 16, 17, 18, 19]

    HAND_VER_0_4_2 = '0.4.2'
    HAND_VER_0_5_1 = '0.5.1'

    _DIO_ASSIGN_OFF = 0
    _DIO_ASSIGN_ON = 1

    def __init__(self):
        """
        Do not get confused that there is also a method called
        'init' (without trailing underscores) that is succeeded from the
        super class as the tradition there.
        """
        super(NextageClient, self).__init__()
        self.set_hand_version(self.HAND_VER_0_5_1)

    def init(self, robotname="HiroNX(Robot)0", url=""):
        """
        Calls init from its superclass, which tries to connect RTCManager,
        looks for ModelLoader, and starts necessary RTC components. Also runs
        config, logger.
        Also internally calls setSelfGroups().

        @end_type robotname: str
        @end_type url: str
        """
        HIRONX.init(self, robotname=robotname, url=url)

    def get_hand_version(self):
        """
        @rtype: str
        """
        if not self._hand_version:
            return 'Hand module not set yet.'
        else:
            return self._hand_version

    def set_hand_version(self, version=HAND_VER_0_5_1):
        self._hand_version = version
        if self.HAND_VER_0_4_2 == self._hand_version:
            self._hands = Iros13Hands(self)
        elif self.HAND_VER_0_5_1 == self._hand_version:
            self._hands = Hands05(self)

    def gripper_l_close(self):
        # return self._gripper_l_command.execute(self._gripper_l_command.GRIPPER_CLOSE)
        return self._hands.gripper_l_close()

    def gripper_r_close(self):
        # return self._gripper_r_command.execute(self._gripper_r_command.GRIPPER_CLOSE)
        return self._hands.gripper_r_close()

    def gripper_l_open(self):
        # return self._gripper_l_command.execute(self._gripper_r_command.GRIPPER_OPEN)
        return self._hands.gripper_l_open()

    def gripper_r_open(self):
        # return self._gripper_r_command.execute(self._gripper_r_command.GRIPPER_OPEN)
        return self._hands.gripper_r_open()

    def handtool_l_eject(self):
        # return self._toolchanger_l_command.execute(
        #     self._toolchanger_l_command.HAND_TOOLCHANGE_OFF)
        return self._hands.handtool_l_eject()

    def handtool_r_eject(self):
        # return self._toolchanger_r_command.execute(
        #     self._toolchanger_r_command.HAND_TOOLCHANGE_OFF)
        return self._hands.handtool_r_eject()

    def handtool_l_attach(self):
        # return self._toolchanger_l_command.execute(
        #     self._toolchanger_l_command.HAND_TOOLCHANGE_ON)
        return self._hands.handtool_l_attach()

    def handtool_r_attach(self):
        # return self._toolchanger_r_command.execute(
        #     self._toolchanger_r_command.HAND_TOOLCHANGE_ON)
        return self._hands.handtool_r_attach()

    def airhand_l_drawin(self):
        return self._hands.airhand_l_drawin()

    def airhand_r_drawin(self):
        return self._hands.airhand_r_drawin()

    def airhand_l_keep(self):
        return self._hands.airhand_l_keep()

    def airhand_r_keep(self):
        return self._hands.airhand_r_keep()

    def airhand_l_release(self):
        return self._hands.airhand_l_release()

    def airhand_r_release(self):
        return self._hands.airhand_r_release()

    def getRTCList(self):
        """
        Overwriting HrpsysConfigurator.getRTCList
        Returning predefined list of RT components.
        @rtype [[str]]
        @return List of available components. Each element consists of a list
                 of abbreviated and full names of the component.
        """
        return [
            ['seq', "SequencePlayer"],
            ['sh', "StateHolder"],
            ['fk', "ForwardKinematics"],
            ['el', "SoftErrorLimiter"],
            # ['co', "CollisionDetector"],
            # ['sc', "ServoController"],
            ['ic', "ImpedanceController"],
            ['log', "DataLogger"]
        ]

    def goInitial(self, tm=7):
        """
        @see: HIRONX.goInitial
        """
        return HIRONX.goInitial(self, tm)

    def readDinGroup(self, ports, dumpflag=True):
        """
        Print the currently set values of digital input registry. Print output order is tailored
        for the hands' functional group; DIO spec that is disloseable as of 7/17/2014 is:

             Left hand:
                  DI26: Tool changer attached or not.
                  DI22, 23: Fingers.
                  DI24, 25: Compliance.

             Right hand:
                  DI21: Tool changer attached or not.
                  DI17, 18: Fingers.
                  DI19, 20: Compliance.

        Example output, for the right hand:

            No hand attached:

                In [1]: robot_s.printDin([20, 16, 17, 18, 19])
                DI21 is 0
                DI17 is 0
                DI18 is 0
                DI19 is 0
                DI20 is 0
                Out[1]: [(20, 0), (16, 0), (17, 0), (18, 0), (19, 0)]

            Hand attached, fingers closed:

                In [1]: robot_s.printDin([20, 16, 17, 18, 19])
                DI21 is 1
                DI17 is 1
                DI18 is 0
                DI19 is 0
                DI20 is 0
                Out[1]: [(20, 0), (16, 0), (17, 0), (18, 0), (19, 0)]

        @author: Koichi Nagashima
        @since: 0.2.16
        @end_type ports: int or [int].
        @param dumpFlag: Print each pin if True.
        @param ports: A port number or a list of port numbers in D-in registry.
        @rtype: [(int, int)]
        @return: List of tuples of port and din value. If the arg ports was an int value,
                 this could be a list with single tuple in it.
        """
        if isinstance(ports, int):
            ports = [ports]
            pass
        # din = self.rh_svc.readDigitalInput()[1];
        ## rh_svc.readDigitalInput() returns tuple, of which 1st element is not needed here.
        din = self.readDigitalInput()
        resary = []
        for port in ports:
            res = din[port]
            if (dumpflag):
                print("DI%02d is %d" % (port + 1, res))
            resary.append((port, res))
            pass
        return resary

    def readDinGroupL(self, dumpflag=True):
        return self.readDinGroup(self._DI_PORTS_L, dumpflag)

    def readDinGroupR(self, dumpflag=True):
        return self.readDinGroup(self._DI_PORTS_R, dumpflag)

    def dioWriter(self, indices, assignments, padding=_DIO_ASSIGN_OFF):
        return self._hands._dio_writer(indices, assignments, padding)

