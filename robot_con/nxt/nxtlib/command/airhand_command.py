"""
revision bassed on nextage_ros_bridge.command.airhand_command by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

import time
import threading

from robotconn.rpc.nxtrobot.nxtlib.command.abs_hand_command import AbsractHandCommand

class AirhandReleaseThread(threading.Thread):
    """
    With airhand, to release by blowing air needs to be stopped after certain
    amount of time. Usually a few seconds, not too short for the air compressor
    to have ample time for a reaction, is a good idea. This thread is used
    in order to do so without letting the computer program halt.
    """
    def __init__(self, command, sleeptime):
        """
        @end_type command: AirhandCommand
        """
        threading.Thread.__init__(self)
        self._command = command
        self._sleeptime = sleeptime

    def run(self):
        time.sleep(self._sleeptime)
        self._command.execute(self._command.AIRHAND_KEEP)


class AirhandCommand(AbsractHandCommand):
    """dio_writer
    Following Command design pattern, this class represents command for
    an Airhand of NEXTAGE OPEN.

    As of 2/1/2014, it's only implemented for a right arm (since there's no
    testing environment for left arm).
    """
    # TODO: Unittest is needed!!

    # For air hands
    AIRHAND_DRAWIN = 'drawin'
    AIRHAND_KEEP = 'keep'
    AIRHAND_RELEASE = 'release'

    ## Might not be necessary. Maybe use only where you have to specify
    ## dangerous situation.AIRHAND_KEEP
    AIRHAND_DANGER = 'danger'

    def __init__(self, hands, hand, dio_pins):
        """
        @see abs_hand_command.AbsractHandCommand
        @end_type hands: base_hands.BaseHands
        @end_type hand: str
        @param hand: Side of hand. Variables that are defined in
        base_hands.BaseHands can be used { HAND_L, HAND_R }.
        """
        super(AirhandCommand, self).__init__(hands, hand, dio_pins)
        self._SLEEP_POST_RELEASE = 3.0

    def _assign_dio_names(self, dio_pins):
        """
        @see abs_hand_command.AbsractHandCommand._assign_dio_names
        """
        #DIO reassignment for the class-specific purpose
        self._DIO_SUCTION_L_1 = dio_pins[0]
        self._DIO_SUCTION_L_2 = dio_pins[1]
        self._DIO_SUCTION_R_1 = dio_pins[2]
        self._DIO_SUCTION_R_2 = dio_pins[3]

    def execute(self, operation):
        """
        @see abs_hand_command.AbsractHandCommand.execute
        """
        dout = []
        mask = []  # Will be filled depending on the side of the hand.
        mask_l = [self._DIO_SUCTION_L_1, self._DIO_SUCTION_L_2]
        mask_r = [self._DIO_SUCTION_R_1, self._DIO_SUCTION_R_2]

        # Set masking value per hand.
        if self._hands.HAND_L == self._hand:
            mask = mask_l
        elif self._hands.HAND_R == self._hand:
            mask = mask_r
        else:
            raise RuntimeError('Make sure _hands object, _hand value are set.')

        if self.AIRHAND_DRAWIN == operation:
            if self._hands.HAND_L == self._hand:
                dout = [self._DIO_SUCTION_L_1]
            elif self._hands.HAND_R == self._hand:
                dout = [self._DIO_SUCTION_R_1]
        elif self.AIRHAND_KEEP == operation:
            if self._hands.HAND_L == self._hand:
                pass  # Do nothing since off for both pins.
            elif self._hands.HAND_R == self._hand:
                pass  # Do nothing since off for both pins.
        elif self.AIRHAND_RELEASE == operation:
            if self._hands.HAND_L == self._hand:
                dout = [self._DIO_SUCTION_L_2]
            elif self._hands.HAND_R == self._hand:
                dout = [self._DIO_SUCTION_R_2]

            # Create a thread to do KEEP action after the specified amount
            # of time without stopping the program.
            thread = AirhandReleaseThread(self, self._SLEEP_POST_RELEASE)
            thread.start()
        else:
            # TODO: Might want to thrown exception?
            return
        return self._hands._dio_writer(dout, mask)
