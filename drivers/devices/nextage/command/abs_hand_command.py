"""
revision bassed on nextage_ros_bridge.command.abs_hand_command by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

class AbsractHandCommand(object):
    """
    Following Command design pattern, this class represents an abstract
    command for hand classes of NEXTAGE OPEN.
    """
    # TODO: Unittest is needed!!DIO_V

    def __init__(self, hands, hand, dio_pins):
        """
        @end_type hands: nextage_ros_bridge.base_hands.BaseHands
        @end_type hand: str
        @param hand: Side of hand. Variables that are defined in
                     nextage_ros_bridge.base_hands.BaseHands can be used
                     { HAND_L, HAND_R }.
        @end_type dio_pins: [int]
        @param dio_pins: List of DIO pins that are used in each HandCommand
                         class. The order is important; it needs be defined
                         in subclasses.
        """
        self._hands = hands
        self._hand = hand
        self._assign_dio_names(dio_pins)

    def execute(self, operation):
        """
        Needs overriddedn, otherwise expcetion occurs.

        @end_type operation: str
        @param operation: name of the operation.
        @rtype: bool
        @return: True if dout was writtable to the register. False otherwise.

        @raise exception: RuntimeError
        """
        msg = 'AbsractHandCommand.execute() not extended.'
        raise NotImplementedError(msg)

    def _assign_dio_names(self, dio_pins):
        """
        It's recommended in the derived classes to re-assign DIO names to
        better represent the specific purposes of each DIO pin in there.
        Since doing so isn' mandatory, this method doesn't emit error even when
        it's not implemented.
        @end_type dio_pins: [int]
        @param dio_pins: List of DIO pins that are used in each HandCommand
                         class. The order is important; it needs be defined
                         in subclasses.
        """
        pass
