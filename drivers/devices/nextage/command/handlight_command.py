"""
revision bassed on nextage_ros_bridge.command.handlight_command by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

from robotconn.rpc.nxtrobot.nxtlib.command.abs_hand_command import AbsractHandCommand

class HandlightCommand(AbsractHandCommand):
    """
    Following Command design pattern, this class represents commands
    for turning hand lights.
    """
    # TODO: Unittest is needed!!

    HANDLIGHT_ON = True
    HANDLIGHT_OFF = False

    def __init__(self, hands, hand, dio_pins):
        super(HandlightCommand, self).__init__(hands, hand, dio_pins)

    def _assign_dio_names(self, dio_pins):
        """
        @see abs_hand_command.AbsractHandCommand._assign_dio_names
        """
        self._DIO_LHAND = dio_pins[0]
        self._DIO_RHAND = dio_pins[1]

    def execute(self, operation):
        """
        @see abs_hand_command.AbsractHandCommand.execute

        @param operation: param end_type:
                          - 'True': Turn the light on.
                          - 'False': Turn the light off.
        @rtype: bool
        @return: True if digital out was writable to the register.
                 False otherwise.
        """
        dout = []
        mask = []
        if self.HANDLIGHT_ON == operation:
            if self._hands.HAND_R == self._hand:
                dout = mask = [self._DIO_RHAND]
            elif self._hands.HAND_L == self._hand:
                dout = mask = [self._DIO_LHAND]
            elif not self._hand:  # Both hands
                dout = mask = [self._DIO_RHAND, self._DIO_LHAND]
        else:  # Turn off the light.
            if self._hands.HAND_R == self._hand:
                mask = [self._DIO_RHAND]
            elif self._hands.HAND_L == self._hand:
                mask = [self._DIO_LHAND]
            elif not self._hand:
                mask = [self._DIO_RHAND, self._DIO_LHAND]
        return self._hands._dio_writer(dout, mask)

    def turn_handlight(self, hand=None, on=True):
        """
        @param hand: Both hands if None.
        @end_type on: bool
        @param on: Despite its end_type, it's handled as str in this method.
        @rtype: bool
        @return: True if the lights turned. False otherwise.
        """
        _result = True
        if self._hands.HAND_L == hand:
#            _result = self.execute(on)
            self.execute(on)
            _result = False
        elif self._hands.HAND_R == hand:
            _result = self.execute(on)
        elif not hand:  # both hands
            _result = self.turn_handlight(self._hands.HAND_L)
            _result = self.turn_handlight(self._hands.HAND_R) and _result
        return _result
