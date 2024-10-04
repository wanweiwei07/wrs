"""
revision bassed on nextage_ros_bridge.command.toolchanger_command by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

from robotconn.rpc.nxtrobot.nxtlib.command.abs_hand_command import AbsractHandCommand

class ToolchangerCommand(AbsractHandCommand):
    """
    Following Command design pattern, this class represents commands for
    a toolchanger of NEXTAGE OPEN.
    """

    # For grippers
    HAND_TOOLCHANGE_ON = 'toolchange_on'
    HAND_TOOLCHANGE_OFF = 'toolchange_off'

    def __init__(self, hands, hand, dio_pins):
        super(ToolchangerCommand, self).__init__(hands, hand, dio_pins)

    def _assign_dio_names(self, dio_pins):
        """
        @see abs_hand_command.AbsractHandCommand._assign_dio_names
        """
        self._DIO_VALVE5PORT_L = dio_pins[0]
        self._DIO_AIR_DRAWIN_L = dio_pins[1]
        self._DIO_AIR_RELEASE_L = dio_pins[2]
        self._DIO_VALVE5PORT_R = dio_pins[3]
        self._DIO_AIR_DRAWIN_R = dio_pins[4]
        self._DIO_AIR_RELEASE_R = dio_pins[5]

    def execute(self, operation):
        """
        @see abs_hand_command.AbsractHandCommand.execute
        """
        dout = []
        # Chuck hand uses 2 bits, either of which needs to remain on to keep
        # grasping position firmly. This becomes an issue when a hand is
        # detatched after some grasping actions where the air keeps blowing
        # out. Thus when detatched, air bits for chuck hand need to be turned
        # off and these 2 bits are included in the masking bit.
        mask = []
        if self.HAND_TOOLCHANGE_ON == operation:
            if self._hands.HAND_L == self._hand:
                # 10/29/2013 DIO changed. Now '1' is ON for both 5PORT Valves.
                mask = [self._DIO_VALVE5PORT_L]
            elif self._hands.HAND_R == self._hand:
                mask = [self._DIO_VALVE5PORT_R]
        elif self.HAND_TOOLCHANGE_OFF == operation:
            if self._hands.HAND_L == self._hand:
                # 10/29/2013 DIO changed. Now '0' is OFF for both 5PORT Valves.
                # 1/31/2014 DIO changed. Now '1' is OFF for both 5PORT Valves.
                mask.append(self._DIO_VALVE5PORT_L)
                mask.append(self._DIO_AIR_DRAWIN_L)
                dout = [self._DIO_VALVE5PORT_L]
            elif self._hands.HAND_R == self._hand:
                mask.append(self._DIO_VALVE5PORT_R)
                mask.append(self._DIO_AIR_DRAWIN_R)
                dout = [self._DIO_VALVE5PORT_R]
        return self._hands._dio_writer(dout, mask)

    def release_ejector(self, hand=None, on=True):
        """
        This function might be used to stop the air flow after ejecting hands.
        It is not usable at 20190419.
        The _DIO_EJECTOR is set to
        _pins_toolchanger = [self.DIO_20, self.DIO_27, self.DIO_28,  # L-hand
                             self.DIO_19, self.DIO_25, self.DIO_26]  # R-hand
        They are for sucking, not blowing.
        Also, the self.HAND_ is not available. They might be self._hand.

        See hands_05.py for details

        author: weiwei
        date: 20190419
        """

        dout = []
        mask = []
        if on:
            if self.HAND_R == hand:
                # TODO: Make sure if turning both ejectors at once is the right
                #      usage.
                dout = mask = [self._DIO_EJECTOR_R_1, self._DIO_EJECTOR_R_2]
            elif self.HAND_L == hand:
                dout = mask = [self._DIO_EJECTOR_L_1, self._DIO_EJECTOR_L_2]
            elif not hand:
                dout = mask = [self._DIO_EJECTOR_R_1, self._DIO_EJECTOR_R_2,
                               self._DIO_EJECTOR_L_1, self._DIO_EJECTOR_L_2]
        else:
            if self.HAND_R == hand:
                mask = [self._DIO_EJECTOR_R_1, self._DIO_EJECTOR_R_2]
            elif self.HAND_L == hand:
                mask = [self._DIO_EJECTOR_L_1, self._DIO_EJECTOR_L_2]
            elif not hand:
                mask = [self._DIO_EJECTOR_R_1, self._DIO_EJECTOR_R_2,
                        self._DIO_EJECTOR_L_1, self._DIO_EJECTOR_L_2]
        return self._hands._dio_writer(dout, mask)
