"""
revision bassed on nextage_ros_bridge.hand_05 by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

from robotconn.rpc.nxtrobot.nxtlib.base_hands import BaseHands
from robotconn.rpc.nxtrobot.nxtlib.command.airhand_command import AirhandCommand
from robotconn.rpc.nxtrobot.nxtlib.command.gripper_command import GripperCommand
from robotconn.rpc.nxtrobot.nxtlib.command.handlight_command import HandlightCommand
from robotconn.rpc.nxtrobot.nxtlib.command.toolchanger_command import ToolchangerCommand

class Hands05(BaseHands):
    """
    This class holds methods to operate the hands of NEXTAGE OPEN, based on the
    specification of the robot_s made by the manufacturer (Kawada)
    in August 2014 and thereafter.

    For the robot_s shipped before then, use Iros13Hands.
    """

    def __init__(self, parent):
        """
        @see: AbsractIros13Hands.__init__

        Digital IOs
        17 right light, 18 left light
        19 right detach, 20 left detach
        21, 22 right blow (right hand tool open/close)
        23, 24 left blow (left hand tool open/close)
        25, 26 right suction (right air hand)
        27, 28 left suction (left air hand)

        author: weiwei, yan
        date: 20190419
        """

        super(Hands05, self).__init__(parent)
        _pins_airhand = [self.DIO_27, self.DIO_28, self.DIO_25, self.DIO_26]
        _pins_gripper = [self.DIO_23, self.DIO_24, self.DIO_21, self.DIO_22]
        _pins_handlight = [self.DIO_18, self.DIO_17]
        _pins_toolchanger = [self.DIO_20, self.DIO_27, self.DIO_28,  # L-hand
                             self.DIO_19, self.DIO_25, self.DIO_26]  # R-hand

        self._airhand_l_command = AirhandCommand(self, self.HAND_L, _pins_airhand)
        self._airhand_r_command = AirhandCommand(self, self.HAND_R, _pins_airhand)
        self._gripper_l_command = GripperCommand(self, self.HAND_L, _pins_gripper)
        self._gripper_r_command = GripperCommand(self, self.HAND_R, _pins_gripper)

        # The following lines are moved from BaseToolchangerHands
        self._handlight_l_command = HandlightCommand(self, self.HAND_L, _pins_handlight)
        self._handlight_r_command = HandlightCommand(self, self.HAND_R, _pins_handlight)
        self._toolchanger_l_command = ToolchangerCommand(self, self.HAND_L, _pins_toolchanger)
        self._toolchanger_r_command = ToolchangerCommand(self, self.HAND_R, _pins_toolchanger)

    def airhand_l_drawin(self):
        return self._airhand_l_command.execute(self._airhand_l_command.AIRHAND_DRAWIN)

    def airhand_r_drawin(self):
        return self._airhand_r_command.execute(self._airhand_r_command.AIRHAND_DRAWIN)

    def airhand_l_keep(self):
        return self._airhand_l_command.execute(self._airhand_l_command.AIRHAND_KEEP)

    def airhand_r_keep(self):
        return self._airhand_r_command.execute(self._airhand_r_command.AIRHAND_KEEP)

    def airhand_l_release(self):
        return self._airhand_l_command.execute(self._airhand_l_command.AIRHAND_RELEASE)

    def airhand_r_release(self):
        return self._airhand_r_command.execute(self._airhand_r_command.AIRHAND_RELEASE)

    def gripper_l_close(self):
        return self._gripper_l_command.execute(self._gripper_l_command.GRIPPER_CLOSE)

    def gripper_r_close(self):
        return self._gripper_r_command.execute(self._gripper_r_command.GRIPPER_CLOSE)

    def gripper_l_open(self):
        return self._gripper_l_command.execute(self._gripper_r_command.GRIPPER_OPEN)

    def gripper_r_open(self):
        return self._gripper_r_command.execute(self._gripper_r_command.GRIPPER_OPEN)

    def handtool_l_eject(self):
        self._dio_writer([], [self.DIO_23, self.DIO_24], self._DIO_ASSIGN_OFF)
        return self._toolchanger_l_command.execute(
            self._toolchanger_l_command.HAND_TOOLCHANGE_OFF)

    def handtool_r_eject(self):
        self._dio_writer([], [self.DIO_21, self.DIO_22], self._DIO_ASSIGN_OFF)
        return self._toolchanger_r_command.execute(
            self._toolchanger_r_command.HAND_TOOLCHANGE_OFF)

    def handtool_l_attach(self):
        return self._toolchanger_l_command.execute(
            self._toolchanger_l_command.HAND_TOOLCHANGE_ON)

    def handtool_r_attach(self):
        return self._toolchanger_r_command.execute(
            self._toolchanger_r_command.HAND_TOOLCHANGE_ON)

    def handlight_r(self, is_on=True):
        return self._handlight_r_command.turn_handlight(self.HAND_R, is_on)

    def handlight_l(self, is_on=True):
        return self._handlight_l_command.turn_handlight(self.HAND_L, is_on)

    def handlight_both(self, is_on=True):
        result = self._handlight_l_command.turn_handlight(self.HAND_L, is_on)
        result = self._handlight_r_command.turn_handlight(self.HAND_R, is_on) and result
        return result
