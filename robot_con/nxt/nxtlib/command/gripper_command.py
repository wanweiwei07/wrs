"""
revision bassed on nextage_ros_bridge.command.gripper_command by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

from robotconn.rpc.nxtrobot.nxtlib.command.abs_hand_command import AbsractHandCommand

class GripperCommand(AbsractHandCommand):
    """
    Following Command design pattern, this class represents an abstract
    command for hand classes of NEXTAGE OPEN.

    NOTE: 1/31/2014 TODO: Only right hand is implemented for now.
    """
    # TODO: Unittest is needed!!

    GRIPPER_CLOSE = 'close'
    GRIPPER_OPEN = 'open'
    GRIPPER_DANGER = 'danger'

    def __init__(self, hands, hand, dio_pins):
        super(GripperCommand, self).__init__(hands, hand, dio_pins)

    def _assign_dio_names(self, dio_pins):
        """
        @see abs_hand_command.AbsractHandCommand._assign_dio_names
        """
        self._DIO_VALVE_L_1 = dio_pins[0]
        self._DIO_VALVE_L_2 = dio_pins[1]
        self._DIO_VALVE_R_1 = dio_pins[2]
        self._DIO_VALVE_R_2 = dio_pins[3]

    def execute(self, operation):
        """
        @see abs_hand_command.AbsractHandCommand.execute
        """
        dout = []
        mask_l = [self._DIO_VALVE_L_1, self._DIO_VALVE_L_2]
        mask_r = [self._DIO_VALVE_R_1, self._DIO_VALVE_R_2]
        if self.GRIPPER_CLOSE == operation:
            if self._hands.HAND_L == self._hand:
                dout = [self._DIO_VALVE_L_1]

            elif self._hands.HAND_R == self._hand:
                dout = [self._DIO_VALVE_R_1]
        elif self.GRIPPER_OPEN == operation:
            if self._hands.HAND_L == self._hand:
                dout = [self._DIO_VALVE_L_2]
            elif self._hands.HAND_R == self._hand:
                dout = [self._DIO_VALVE_R_2]
        mask = None 
        if self._hands.HAND_L == self._hand:
            mask = mask_l
        elif self._hands.HAND_R == self._hand:
            mask = mask_r
        return self._hands._dio_writer(dout, mask)
