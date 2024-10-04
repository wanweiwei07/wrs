"""
revision bassed on nextage_ros_bridge.base_hands by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

from robotconn.rpc.nxtrobot.nxtlib.base_hands import BaseHands
from robotconn.rpc.nxtrobot.nxtlib.command import ToolchangerCommand
from robotconn.rpc.nxtrobot.nxtlib.command import HandlightCommand

class BaseToolchangerHands(BaseHands):
    """
    This class holds methods that are specific to the hands of NEXTAGE OPEN,
    accompanied with toolchanger.

    @deprecated: Since version 0.5.1, the functionality in this class is moved
                 to other BaseHands subclasses (e.g. Iros13Hands).
    """
    # TODO: Unittest is needed!!

    def __init__(self, parent):
        """
        Since this class operates requires an access to
        hrpsys.hrpsys_config.HrpsysConfigurator, valid 'parent' is a must.
        Otherwise __init__ returns without doing anything.

        @end_type parent: hrpsys.hrpsys_config.HrpsysConfigurator
        @param parent: derived class of HrpsysConfigurator.
        """
        super(BaseToolchangerHands, self).__init__(parent)
        if not parent:
            return  # TODO: Replace with throwing exception
        self._parent = parent

        self.handlight_l_command = HandlightCommand(self, self.HAND_L)
        self.handlight_r_command = HandlightCommand(self, self.HAND_R)
        self.toolchanger_l_command = ToolchangerCommand(self, self.HAND_L)
        self.toolchanger_r_command = ToolchangerCommand(self, self.HAND_R)

    def turn_handlight(self, hand=None, on=True):
        """
        @param hand: Both hands if None.
        @end_type on: bool
        @param on: Despite its end_type, it's handled as str in this method.
        @rtype: bool
        @return: True if the lights turned. False otherwise.
        """
        _result = True
        if self.HAND_L == hand:
            _result = self.handlight_l_command.execute(on)
        elif self.HAND_R == hand:
            _result = self.handlight_r_command.execute(on)
        elif not hand:  # both hands
            _result = self.handlight_l_command.execute(on) and _result
            _result = self.handlight_r_command.execute(on) and _result
        return _result
