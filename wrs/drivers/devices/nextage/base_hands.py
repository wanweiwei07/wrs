"""
revision bassed on nextage_ros_bridge.base_hands by JSK, UTokyo

author: weiwei, yan, osaka
date: 20190417
"""

class BaseHands(object):
    '''
    This class provides methods that are generic for the hands of
    Kawada Industries' dual-arm robot_s called Nextage Open.
    '''
    # TODO: Unittest is needed!!

    # Since NEXTAGE is expected to be dual-arm, arm indicator can be defined
    # at the top level of hierarchy.
    HAND_L = '1'  # '0' is expected to be "Both hands".
    HAND_R = '2'

    _DIO_ASSIGN_ON = 1
    _DIO_ASSIGN_OFF = 0
    _DIO_MASK = 0   # Masking value remains "0" regardless the design on the
                    # robot_s; masking logic is defined in hrpsys, while robot_s
                    # makers can decide DIO logic.

    # DIO pin numbers. It's convenient to be overridden and renamed in the
    # derived classes to represent the specific purpose of each pin.
    DIO_17 = 17
    DIO_18 = 18
    DIO_19 = 19
    DIO_20 = 20
    DIO_21 = 21
    DIO_22 = 22
    DIO_23 = 23
    DIO_24 = 24
    DIO_25 = 25
    DIO_26 = 26
    DIO_27 = 27
    DIO_28 = 28

    _MSG_ERR_NOTIMPLEMENTED = 'The method is not implemented in the derived class'

    def __init__(self, parent):
        '''
        Since this class operates requires an access to
        hrpsys.hrpsys_config.HrpsysConfigurator, valid 'parent' is a must.
        Otherwise __init__ returns without doing anything.

        @end_type parent: hrpsys.hrpsys_config.HrpsysConfigurator
        @param parent: derived class of HrpsysConfigurator.
        '''
        if not parent:
            return  # TODO: Replace with throwing exception
        self._parent = parent

    def _dio_writer(self, digitalout_indices, dio_assignments,
                    padding=_DIO_ASSIGN_OFF):
        '''
        This private method calls HrpsysConfigurator.writeDigitalOutputWithMask,
        which this class expects to be available via self._parent.

        According to the current (Oct 2013) hardware spec, numbering rule
        differs regarding 0 (numeric figure) in dout and mask as follows:

           * 0 is "OFF" in the digital output.
             * 2/1/2014 Assignment modified 0:ON --> 0:OFF
           * 0 is "masked" and not used in mask. Since using '0' is defined in
             hrpsys and not in the robots side, we'll always use '0' for
             masking.

        @end_type digitalout_indices: int[]
        @param digitalout_indices: Array of indices of digital output that NEED to be
                            flagged as 1.
                            eg. If you're targetting on 25 and 26th places in
                                the DIO array but only 25th is 1, then the
                                array becomes [24].
        @end_type dio_assignments: int[]
        @param dio_assignments: range(32). Also called as "masking bits" or
                                just "mask". This number corresponds to the
                               assigned digital pin of the robot_s.

                               eg. If the target pins are 25 and 26,
                                   dio_assignments = [24, 25]
        @param padding: Either 0 or 1. DIO bit array will be populated with
                        this value.
                        Usually this method assumes to be called when turning
                        something "on". Therefore by default this value is ON.
        @rtype: bool
        @return: True if dout was writable to the register. False otherwise.
        '''

        # 32 bit arrays used in write methods in hrpsys/hrpsys_config.py
        dout = []
        for i in range(32):
            dout.append(padding)
            # At the end_type of this loop, dout contains list of 32 'padding's.
            # eg. [ 0, 0,...,0] if padding == 0
        mask = []
        for i in range(32):
            mask.append(self._DIO_MASK)
            # At the end_type of this loop, mask contains list of 32 '0's.

        signal_alternate = self._DIO_ASSIGN_ON
        if padding == self._DIO_ASSIGN_ON:
            signal_alternate = self._DIO_ASSIGN_OFF
        # Assign commanded bits
        for i in digitalout_indices:
            dout[i - 1] = signal_alternate

        # Assign unmasked, effective bits
        for i in dio_assignments:
            # For masking, alternate symbol is always 1 regarless the design
            # on robot_s's side.
            mask[i - 1] = 1

        # BEGIN; For convenience only; to show array number.
        print_index = []
        for i in range(10):
            # For masking, alternate symbol is always 1.
            n = i + 1
            if 10 == n:
                n = 0
            print_index.append(n)
        print_index.extend(print_index)
        print_index.extend(print_index)
        del print_index[-8:]
        # END; For convenience only; to show array number.

        # # For some reason rospy.loginfo not print anything.
        # # With this print formatting, you can copy the output and paste
        # # directly into writeDigitalOutputWithMask method if you wish.
        print('dout, mask:\n{},\n{}\n{}'.format(dout, mask, print_index))

        is_written_dout = False
        try:
            is_written_dout = self._parent.writeDigitalOutputWithMask(dout,
                                                                      mask)
        except AttributeError as e:
            pass
        return is_written_dout

    def init_dio(self):
        '''
        Initialize dio. All channels will be set '0' (off), EXCEPT for
        tool changers (channel 19 and 24) so that attached tools won't fall.
        '''
        # TODO: The behavior might not be optimized. Ask Hajime-san and
        #       Nagashima-san to take a look.

        # 10/24/2013 OUT19, 24 are alternated; When they turned to '1', they
        # are ON. So hands won't fall upon this function call.
        # 2/1/2014 Due to the change of DIO assingment, OUT19, 24 need to be
        # alternated;

        dout = mask = []
        # Use all slots from 17 to 32.
        for i in range(16, 32):
            mask.append(i)

        self._dio_writer(dout, mask, self._DIO_ASSIGN_ON)

    # The following are common hand commands set by default.
    # Depending on the configuration some / all of these don't necessarily
    # have to be implemented.
    def airhand_l_drawin(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def airhand_r_drawin(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def airhand_l_keep(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def airhand_r_keep(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def airhand_l_release(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def airhand_r_release(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def gripper_l_close(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def gripper_r_close(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def gripper_l_open(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def gripper_r_open(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handlight_r(self, is_on=True):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handlight_l(self, is_on=True):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handlight_both(self, is_on=True):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handtool_l_eject(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handtool_r_eject(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handtool_l_attach(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)

    def handtool_r_attach(self):
        raise NotImplementedError(self._MSG_ERR_NOTIMPLEMENTED)
