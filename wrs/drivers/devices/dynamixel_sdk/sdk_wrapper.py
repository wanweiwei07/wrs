"""
Control Dynamixel Motor Through XArm Ethernet Pass-Through Communication
Author: Chen Hao (chen960216@gmail.com), 20220915, osaka
Reference: XArm Developer Manual (http://download.ufactory.cc/xarm/en/xArm%20Developer%20Manual.pdf?v=1600992000052)
           Dynamixel Protocol 2.0 (https://emanual.robotis.com/docs/en/dxl/protocol2/)
           Dynamixel XM430 Manual (https://emanual.robotis.com/docs/en/dxl/x/xm430-w350/)
Update Notes:
    -`0.0.1`: Implement the basic function (Ping, Enable/Disable Torque, Set/Get Motor Position)
    -`0.0.2`: Add functions to access the velocity, current, and operation mode
    -`0.0.3` 20231008: Wrap with high-level API

TODO:
    - Currently, sync write is only support for the goal postion. Add more support in future
    - Test Linux support
"""
import time
import importlib
from typing import Literal
from collections import namedtuple
from wrs.drivers.devices.dynamixel_sdk.protocol2_packet_handler import (Protocol2PacketHandler,
                                                                        COMM_SUCCESS,
                                                                        DXL_LOBYTE,
                                                                        DXL_HIBYTE,
                                                                        DXL_LOWORD,
                                                                        DXL_HIWORD, )
from wrs.drivers.devices.dynamixel_sdk.group_sync_write import GroupSyncWrite


def import_port_handler(module_path: str):
    try:
        # Examine if the module_path is valid
        if not isinstance(module_path, str):
            raise TypeError(f"The model path '{module_path}' is not a string.")

        # Use importlib to import the module dynamically
        imported_module = importlib.import_module(module_path)

        # Assuming your special model class is named 'SpecialModel' in the module
        if hasattr(imported_module, 'PortHandler'):  # Examine if the module contains the special model
            return imported_module.PortHandler
        else:
            raise AttributeError(f"The module '{module_path}' does not contain 'SpecialModel' class.")
    except ImportError:
        raise ImportError(f"Could not import the special model '{module_path}'")


PortHandler = import_port_handler('drivers.devices.dynamixel_sdk.port_handler')

__VERSION__ = '0.0.3'

LATENCY_TIMER = 16
DXL_CONTROL_TABLE = namedtuple("DXL_CONTROL_TABLE", ['ADDR_OPERATION_MODE',
                                                     'ADDR_TORQUE_ENABLE',
                                                     'ADDR_GOAL_POSITION',
                                                     'ADDR_GOAL_CURRENT',
                                                     'ADDR_GOAL_VELOCITY',
                                                     'ADDR_MOVING',
                                                     'ADDR_MOVING_STATUS',
                                                     'ADDR_CURRENT_LIMIT',
                                                     'ADDR_PRESENT_POSITION',
                                                     'ADDR_PRESENT_CURRENT',
                                                     'ADDR_PRESENT_VELOCITY',
                                                     'ADDR_LED_RED',
                                                     'ADDR_POSITION_P_GAIN',
                                                     'DXL_MIN_POSITION_VAL',
                                                     'DXL_MAX_POSITION_VAL',
                                                     'DXL_MIN_CURRENT_VAL',
                                                     'DXL_MAX_CURRENT_VAL',
                                                     'DXL_MIN_VELOCITY_VAL',
                                                     'DXL_MAX_VELOCITY_VAL',
                                                     'DXL_MIN_POSITION_P_GAIN_VAL',
                                                     'DXL_MAX_POSITION_P_GAIN_VAL',
                                                     ])

CONTROL_TABLE = {
    'X_SERIES': DXL_CONTROL_TABLE(ADDR_OPERATION_MODE=11,
                                  ADDR_TORQUE_ENABLE=64,
                                  ADDR_GOAL_POSITION=116,
                                  ADDR_GOAL_CURRENT=102,
                                  ADDR_GOAL_VELOCITY=104,
                                  ADDR_MOVING=122,
                                  ADDR_MOVING_STATUS=123,
                                  ADDR_CURRENT_LIMIT=38,
                                  ADDR_PRESENT_POSITION=132,
                                  ADDR_PRESENT_CURRENT=126,
                                  ADDR_PRESENT_VELOCITY=128,
                                  ADDR_LED_RED=65,
                                  DXL_MIN_POSITION_VAL=0,
                                  DXL_MAX_POSITION_VAL=4095,  # 0.088deg per unit
                                  DXL_MIN_CURRENT_VAL=0,
                                  DXL_MAX_CURRENT_VAL=1193,  # 2.89mA per unit
                                  DXL_MIN_VELOCITY_VAL=0,
                                  DXL_MAX_VELOCITY_VAL=1023,  # 0.229rpm=0.024rad/s per unit
                                  ADDR_POSITION_P_GAIN=84,
                                  DXL_MIN_POSITION_P_GAIN_VAL=0,
                                  DXL_MAX_POSITION_P_GAIN_VAL=16383,
                                  )
}


class DynamixelMotor(object):
    """
     Class for communication for the Dynamixel motor through RS485 based on Dynamixel Protocol 2.0
     Reference: https://emanual.robotis.com/docs/en/dxl/protocol2/
     Note: Set the torque option of the motor ON. Otherwise, the communication will be error
     """

    def __init__(self,
                 device='COM6',
                 baud_rate=9600,
                 dxl_mdl: Literal['X_SERIES'] = 'X_SERIES',
                 port_handler: PortHandler = None,
                 packet_handler: Protocol2PacketHandler = None,
                 toggle_group_sync_write: bool = False, ):

        assert dxl_mdl in CONTROL_TABLE
        if port_handler is None:
            self._port_handler = PortHandler(device)
        else:
            self._port_handler = port_handler
        if packet_handler is None:
            self._packet_handler = Protocol2PacketHandler()
        else:
            self._packet_handler = packet_handler

        if toggle_group_sync_write:
            self._group_sync_write = GroupSyncWrite(port=self._port_handler,
                                                    ph=self._packet_handler,
                                                    start_address=CONTROL_TABLE[dxl_mdl].ADDR_GOAL_POSITION,
                                                    data_length=4)
        else:
            self._group_sync_write = None

        # motor information
        self._dxl_mdl = dxl_mdl
        self._control_table: CONTROL_TABLE = CONTROL_TABLE[self._dxl_mdl]

        # set up the baudrate
        self.set_baud_rate(baud_rate=baud_rate)
        self.baud_rate = baud_rate

    def _ex_ret_code(self, code):
        """
        Examine the return code of the instruction. If the code is not 0 (success), a Exception will be raised.
        :param code:
        :return:
        """
        if code != 0:
            raise Exception(f"The return code {code} is incorrect. Refer API for details")

    def set_baud_rate(self, baud_rate):
        """
        Set the baudrate of the motor
        :param baud_rate: the baudrate of the motor
        """
        self._port_handler.setBaudRate(baud_rate)
        self.baud_rate = baud_rate

    def ping(self, dxl_id: int) -> bool:
        """
        Ping the motor to check if the motor is connected
        :param dxl_id: the id of the motor
        :return: True if the motor is connected, False otherwise
        """
        dxl_model_number, dxl_comm_result, dxl_error = self._packet_handler.ping(port=self._port_handler,
                                                                                 dxl_id=dxl_id)
        if dxl_comm_result != COMM_SUCCESS:
            return False
        elif dxl_error != 0:
            return True
        else:
            return False

    def enable_dxl_torque(self, dxl_id: int) -> bool:
        """
        Enable the torque of the motor
        :param dxl_id: the id of the motor
        :return: True if the torque is enabled, False otherwise
        """
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_TORQUE_ENABLE,
                                                                         data=1)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            print("Dynamixel has been successfully connected")
            return True

    def disable_dxl_torque(self, dxl_id: int) -> bool:
        """
        Disable the torque of the motor
        :param dxl_id: the id of the motor
        :return: True if the torque is disabled, False otherwise
        """
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_TORQUE_ENABLE,
                                                                         data=0)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def set_dxl_goal_pos(self, tgt_pos: int = 0, dxl_id: int = 1) -> bool:
        """
        Set the goal position of the motor
        :param tgt_pos: the target position of the motor
        :param dxl_id: the id of the motor
        :return: True if the goal position is set, False otherwise
        """
        assert isinstance(tgt_pos, int)
        assert self._control_table.DXL_MIN_POSITION_VAL <= tgt_pos <= self._control_table.DXL_MAX_POSITION_VAL
        dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(port=self._port_handler,
                                                                         dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_GOAL_POSITION,
                                                                         data=tgt_pos)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def set_dxl_goal_pos_sync(self, tgt_pos_list: list or tuple, dxl_id_list: list or tuple) -> bool:
        if self._group_sync_write is None:
            raise Exception(
                "The group sync write is not enabled. Please set toggle_group_sync_write to True when initializing the class.")
        assert isinstance(tgt_pos_list, (list, tuple)) and isinstance(dxl_id_list, (
            list, tuple)), "The input should be a list or tuple."
        assert len(tgt_pos_list) == len(
            dxl_id_list), "The length of the target position list and the motor id list should be the same."
        for tgt_pos in tgt_pos_list:
            assert isinstance(tgt_pos, int), "The target position should be an integer."
            assert self._control_table.DXL_MIN_POSITION_VAL <= tgt_pos <= self._control_table.DXL_MAX_POSITION_VAL, "The target position is out of range."
        for i, dxl_id in enumerate(dxl_id_list):
            assert isinstance(dxl_id, int), "The motor id should be an integer."
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(tgt_pos_list[i])),
                                   DXL_HIBYTE(DXL_LOWORD(tgt_pos_list[i])),
                                   DXL_LOBYTE(DXL_HIWORD(tgt_pos_list[i])),
                                   DXL_HIBYTE(DXL_HIWORD(tgt_pos_list[i]))]

            dxl_comm_result = self._group_sync_write.addParam(dxl_id, param_goal_position)
            if dxl_comm_result != True:
                print("The group sync write failed.")
                # Clear syncwrite parameter storage
                self._group_sync_write.clearParam()
                return False
        # Syncwrite goal position
        dxl_comm_result = self._group_sync_write.txPacket()
        self._group_sync_write.clearParam()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        else:
            return True

    def get_dxl_goal_pos(self, dxl_id: int) -> int:
        """
        Get the goal position of the motor
        :param dxl_id: the id of the motor
        :return: the goal position of the motor
        """
        dxl_present_position, dxl_comm_result, dxl_error = self._packet_handler.read4ByteTxRx(port=self._port_handler,
                                                                                              dxl_id=dxl_id,
                                                                                              address=self._control_table.ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_position

    def enable_led(self, dxl_id: int) -> bool:
        """
        Enable the LED of the motor
        :param dxl_id: the id of the motor
        :return: True if the LED is enabled, False otherwise
        """
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_LED_RED,
                                                                         data=1)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def disable_led(self, dxl_id: int) -> bool:
        """
        Disable the LED of the motor
        :param dxl_id: the id of the motor
        :return: True if the LED is disabled, False otherwise
        """
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler,
                                                                         dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_LED_RED,
                                                                         data=0)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def get_dxl_op_mode(self, dxl_id: int):
        """
        Get the operation mode of the motor
        :param dxl_id: the id of the motor
        :return: the operation mode of the motor
                    0	Current Control Mode
                    1	Velocity Control Mode
                    3	Position Control Mode (default)
                    4   Extended Position Control Mode (Multi-turn)
                    5   Current-based Position Control Mode
                    16  PWM Control Mode
        """
        dxl_present_mode, dxl_comm_result, dxl_error = self._packet_handler.read1ByteTxRx(port=self._port_handler,
                                                                                          dxl_id=dxl_id,
                                                                                          address=self._control_table.ADDR_OPERATION_MODE)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_mode

    def set_dxl_op_mode(self, op_mode: int, dxl_id: int):
        """
        Set the operation mode of the motor
        [Note] The operation mode should be set BEFORE the motor is enabled.
        :param op_mode: the operation mode of the motor
                    0	Current Control Mode
                    1	Velocity Control Mode
                    3	Position Control Mode (default)
                    4   Extended Position Control Mode (Multi-turn)
                    5   Current-based Position Control Mode
                    16  PWM Control Mode
        :param dxl_id: the id of the motor
        :return: True if the operation mode is set, False otherwise
        """
        assert op_mode in [0, 1, 3, 4, 5, 16]
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler,
                                                                         dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_OPERATION_MODE,
                                                                         data=op_mode)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def set_dxl_goal_vel(self, tgt_vel: int = 0) -> bool:
        """
        Set the goal velocity of the motor
        :param tgt_vel: the target velocity of the motor
        :return: True if the goal velocity is set, False otherwise
        """
        raise NotImplementedError

    def set_dxl_goal_current(self, tgt_current: int, dxl_id: int):
        """
        Set the goal current of the motor
        :param tgt_current: the target current of the motor
        :param dxl_id: the id of the motor
        :return: True if the goal current is set, False otherwise
        """
        assert isinstance(tgt_current, int)
        assert self._control_table.DXL_MIN_CURRENT_VAL <= tgt_current <= self._control_table.DXL_MAX_CURRENT_VAL
        dxl_comm_result, dxl_error = self._packet_handler.write2ByteTxRx(port=self._port_handler,
                                                                         dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_GOAL_CURRENT,
                                                                         data=tgt_current)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def set_dxl_current_limit(self, current_limit: int, dxl_id: int):
        """
        Set the current limit of the motor.
        [Note] The current limit should be set AFTER the motor is enabled.
        :param current_limit: the current limit of the motor
        :param dxl_id: the id of the motor
        :return: True if the current limit is set, False otherwise
        """
        assert isinstance(current_limit, int)
        assert self._control_table.DXL_MIN_CURRENT_VAL <= current_limit <= self._control_table.DXL_MAX_CURRENT_VAL
        dxl_comm_result, dxl_error = self._packet_handler.write2ByteTxRx(port=self._port_handler,
                                                                         dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_CURRENT_LIMIT,
                                                                         data=current_limit)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def get_dxl_pos(self, dxl_id: int) -> int:
        """
        Get the position of the motor
        :param dxl_id: the id of the motor
        :return: the position of the motor
        """
        dxl_present_pos, dxl_comm_result, dxl_error = self._packet_handler.read4ByteTxRx(port=self._port_handler,
                                                                                         dxl_id=dxl_id,
                                                                                         address=self._control_table.ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_pos

    def get_dxl_vel(self, dxl_id: int) -> int:
        """
        Get the velocity of the motor
        :param dxl_id: the id of the motor
        :return: the velocity of the motor
        """
        dxl_present_velocity, dxl_comm_result, dxl_error = self._packet_handler.read4ByteTxRx(port=self._port_handler,
                                                                                              dxl_id=dxl_id,
                                                                                              address=self._control_table.ADDR_PRESENT_VELOCITY)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_velocity

    def get_dxl_current(self, dxl_id: int) -> int:
        """
        Get the current of the motor
        :param dxl_id: the id of the motor
        :return: the current of the motor
        """
        dxl_present_current, dxl_comm_result, dxl_error = self._packet_handler.read2ByteTxRx(port=self._port_handler,
                                                                                             dxl_id=dxl_id,
                                                                                             address=self._control_table.ADDR_PRESENT_CURRENT)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_current

    def get_dxl_current_limit(self, dxl_id: int) -> int:
        """
        Get the current limit of the motor
        :param dxl_id: the id of the motor
        :return: the current limit of the motor
        """
        dxl_present_current, dxl_comm_result, dxl_error = self._packet_handler.read2ByteTxRx(port=self._port_handler,
                                                                                             dxl_id=dxl_id,
                                                                                             address=self._control_table.ADDR_CURRENT_LIMIT)

        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_current

    def is_moving(self, dxl_id) -> bool:
        """
        Check if the motor is moving
        :param dxl_id: the id of the motor
        :return: True if the motor is moving, False otherwise
        """
        dxl_is_moving, dxl_comm_result, dxl_error = self._packet_handler.read1ByteTxRx(port=self._port_handler,
                                                                                       dxl_id=dxl_id,
                                                                                       address=self._control_table.ADDR_MOVING)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            raise Exception("Communication Error")
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            raise Exception("Communication Error")
        else:
            return dxl_is_moving

    def get_moving_status(self, dxl_id: int) -> int:
        """
        Get the moving status of the motor
        :param dxl_id: the id of the motor
        :return: the moving status of the motor (See the reference for details)
        """
        dxl_moving_status, dxl_comm_result, dxl_error = self._packet_handler.read1ByteTxRx(port=self._port_handler,
                                                                                           dxl_id=dxl_id,
                                                                                           address=self._control_table.ADDR_MOVING_STATUS)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            raise Exception("Communication Error")
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            raise Exception("Communication Error")
        else:
            return dxl_moving_status

    def set_dxl_position_p_gain(self, p_gain_val: int, dxl_id: int) -> bool:
        """
        Set the position P gain of the motor. (Related to the velocity)
        :param p_gain_val: the position P gain of the motor
        :param dxl_id: the id of the motor
        :return: True if the position P gain is set, False otherwise
        """
        assert isinstance(p_gain_val, int)
        assert self._control_table.DXL_MIN_CURRENT_VAL <= p_gain_val <= self._control_table.DXL_MAX_CURRENT_VAL
        dxl_comm_result, dxl_error = self._packet_handler.write2ByteTxRx(port=self._port_handler,
                                                                         dxl_id=dxl_id,
                                                                         address=self._control_table.ADDR_POSITION_P_GAIN,
                                                                         data=p_gain_val)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return False
        elif dxl_error != 0:
            print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return False
        else:
            return True

    def get_dxl_position_p_gain(self, dxl_id: int) -> int:
        """
        Get the position P gain of the motor. (Related to the velocity)
        :param dxl_id: the id of the motor
        :return: the position P gain of the motor
        """
        dxl_position_p_gain, dxl_comm_result, dxl_error = self._packet_handler.read2ByteTxRx(port=self._port_handler,
                                                                                             dxl_id=dxl_id,
                                                                                             address=self._control_table.ADDR_POSITION_P_GAIN, )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_position_p_gain


if __name__ == "__main__":

    peripheral_baud = 115200
    com = 'COM7'
    dxl_con = DynamixelMotor(com, baud_rate=peripheral_baud)
    # r = dxl_con.ping()
    # print(dxl_con.get_dxl_position_p_gain())
    # print(dxl_con.get_dxl_op_mode(1))
    # print(dxl_con.get_dxl_op_mode(2))

    # dxl_con.disable_dxl_torque(1)
    # dxl_con.disable_dxl_torque(2)
    # print(dxl_con.get_dxl_pos(1))
    # print(dxl_con.get_dxl_pos(2))
    # exit(0)

    # print(dxl_con.get_dxl_goal_pos())
    # # print(dxl_con.get_dxl_vel())
    # print("is moving:", dxl_con.is_moving())
    # # print(r)

    # 1341  2252
    # 3083 4012

    # # exit(0)
    # dxl_con.set_dxl_position_p_gain(300, 1)
    print("?")
    dxl_con.set_dxl_current_limit(400, 1)
    print("!")
    dxl_con.set_dxl_current_limit(400, 2)
    print("!")
    time.sleep(1)
    print("?")
    print(dxl_con.get_dxl_current_limit(1), dxl_con.get_dxl_current_limit(2))
    print("?")
    dxl_con.set_dxl_op_mode(5, 1)
    dxl_con.set_dxl_op_mode(5, 2)
    time.sleep(.2)
    dxl_con.enable_dxl_torque(1)
    dxl_con.enable_dxl_torque(2)
    dxl_con.set_dxl_goal_current(160, 1)
    dxl_con.set_dxl_goal_current(160, 2)
    time.sleep(1)
    # dxl_con.set_dxl_goal_pos(20)

    # a = dxl_con.set_dxl_goal_current(20)
    # dxl_con.set_dxl_position_p_gain(50)
    # print("Set torque", a)
    # print(a)
    # dxl_con.set_dxl_goal_pos(1000)
    # # print("is moving:", dxl_con.is_moving())
    # time.sleep(2)
    # dxl_con.set_dxl_goal_pos(389)
    # time.sleep(2)
    # dxl_con.set_dxl_goal_current(100, 1)
    while 1:
        v = 1500
        s1 = 1373 + v
        s2 = 1112 + v
        # dxl_con.set_dxl_position_p_gain(100, 1)
        # dxl_con.set_dxl_position_p_gain(100, 2)
        dxl_con.set_dxl_position_p_gain(50, 1)
        dxl_con.set_dxl_position_p_gain(50, 2)
        dxl_con.set_dxl_goal_current(80, 1)
        dxl_con.set_dxl_goal_current(80, 2)
        dxl_con.set_dxl_goal_pos(s1, 1)
        dxl_con.set_dxl_goal_pos(s2, 2)
        print(dxl_con.get_dxl_current(1), dxl_con.get_dxl_current(2))
        # exit(0)
        input('b')

        v = -1000
        s1 = 1373 + v
        s2 = 1112 + v
        dxl_con.set_dxl_goal_current(170, 1)
        dxl_con.set_dxl_goal_current(170, 2)
        dxl_con.set_dxl_position_p_gain(20, 1)
        dxl_con.set_dxl_position_p_gain(20, 2)
        dxl_con.set_dxl_goal_pos(s1, 1)
        dxl_con.set_dxl_goal_pos(s2, 2)

        input('a')

    # dxl_con.disable_dxl_torque(1)
    # dxl_con.disable_dxl_torque(2)
