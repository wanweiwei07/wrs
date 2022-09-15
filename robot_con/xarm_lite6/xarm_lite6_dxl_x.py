"""
Control Dynamixel Motor Through XArm Ethernet Pass-Through Communication
Author: Chen Hao (chen960216@gmail.com), 20220915, osaka
Reference: XArm Developer Manual (http://download.ufactory.cc/xarm/en/xArm%20Developer%20Manual.pdf?v=1600992000052)
           Dynamixel Protocol 2.0 (https://emanual.robotis.com/docs/en/dxl/protocol2/)
"""
import time
import socket
from typing import Literal
from collections import namedtuple

from drivers.devices.dynamixel_sdk.protocol2_packet_handler import (Protocol2PacketHandler,
                                                                    COMM_SUCCESS)

__VERSION__ = '0.0.1'

LATENCY_TIMER = 16
DXL_CONTROL_TABLE = namedtuple("DXL_CONTROL_TABLE", ['ADDR_TORQUE_ENABLE',
                                                     'ADDR_GOAL_POSITION',
                                                     'ADDR_PRESENT_POSITION',
                                                     'ADDR_LED_RED',
                                                     'DXL_MINIMUM_POSITION_VALUE',
                                                     'DXL_MAXIMUM_POSITION_VALUE'])


class PortHandler(object):
    """
    PortHandler for Dynamixel SDK PacketHandler. Transport data through 502 port
    """

    def __init__(self, ip, port=502):
        # setup socket
        self._sock = socket.socket()
        self._sock.connect((ip, port))

        self.packet_start_time = 0.0
        self.packet_timeout = 0.0
        self.tx_time_per_byte = 0.0

        self._buffer = []

        self.is_using = False

    @staticmethod
    def pack_data(peripheral_data):
        function_code = 0xF1
        tgpio_id = 0x09
        data_len = len(peripheral_data) + 1 + 1
        # print("data_len:",data_len)
        txdata = bytes([0x00, 0x01, 0x00, 0x02, 0x00])
        txdata += bytes([data_len])
        txdata += bytes([function_code])
        txdata += bytes([tgpio_id])
        txdata += bytes(peripheral_data)
        return txdata

    # useless function
    def clearPort(self):
        del self._buffer
        self._buffer = []

    def writePort(self, packet):
        return self._sock.send(self.pack_data(packet)) - 8

    def readPort(self, _):
        try:
            data = self._sock.recv(100)
            # for print debug info
            # print("read data is", list(data)[9:])
            ret_data = list(data)[9:]
        except Exception as e:
            ret_data = []
            # print(f"[Error] {e}")
        self._buffer.extend(ret_data)
        return self._buffer

    def isPacketTimeout(self):
        if self.getTimeSinceStart() > self.packet_timeout:
            self.packet_timeout = 0
            return True

        return False

    def getTimeSinceStart(self):
        time_since = self.getCurrentTime() - self.packet_start_time
        if time_since < 0.0:
            self.packet_start_time = self.getCurrentTime()

        return time_since

    def setPacketTimeout(self, packet_length):
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (self.tx_time_per_byte * packet_length) + (LATENCY_TIMER * 2.0) + 2.0

    def getCurrentTime(self):
        return round(time.time() * 1000000000) / 1000000.0

    def close(self):
        self._sock.close()

    def __del__(self):
        self.close()


class XArmLite6DXLCon(object):
    """
     Class for communication for the Dynamixel motor through RS485 based on Dynamixel Protocol 2.0
     Reference: https://emanual.robotis.com/docs/en/dxl/protocol2/
     Note: Set the torque option of the motor ON. Otherwise, the communication will be error
     """
    CONTROL_TABLE = {
        'X_SERIES': DXL_CONTROL_TABLE(ADDR_TORQUE_ENABLE=64, ADDR_GOAL_POSITION=116,
                                      ADDR_PRESENT_POSITION=132,
                                      ADDR_LED_RED=65,
                                      DXL_MINIMUM_POSITION_VALUE=0,
                                      DXL_MAXIMUM_POSITION_VALUE=4095)
    }

    def __init__(self, arm_x: 'XArmAPI', baudrate=9600, dxl_id=2, dxl_mdl: Literal['X_SERIES'] = 'X_SERIES'):

        assert dxl_mdl in self.CONTROL_TABLE

        self._arm_x = arm_x
        self._ip = arm_x.arm._port
        self._port_handler = PortHandler(ip=self._ip)
        self._packet_handler = Protocol2PacketHandler()

        # motor information
        self._dxl_id = dxl_id
        self._dxl_mdl = dxl_mdl
        self._control_table = self.CONTROL_TABLE[self._dxl_mdl]

        # set up the baudrate
        self.set_baudrate(baudrate=baudrate)

    @property
    def baudrate(self):
        code, ret = self._arm_x.get_tgpio_modbus_baudrate()
        self._ex_ret_code(code)
        if ret < 0:
            raise Exception("Error occur when acquiring baud rate")
        return ret

    def _ex_ret_code(self, code):
        """
        Examine the return code of the instruction. If the code is not 0 (success), a Exception will be raised.
        :param code:
        :return:
        """
        if code != 0:
            raise Exception(f"The return code {code} is incorrect. Refer API for details")

    def set_baudrate(self, baudrate):
        if self.baudrate != baudrate:
            suc = self._arm_x.set_tgpio_modbus_baudrate(baudrate)
            return suc == 0
        else:
            print(f"Baudrate has already been {baudrate}")
            return True

    def ping(self) -> bool:
        dxl_model_number, dxl_comm_result, dxl_error = self._packet_handler.ping(port=self._port_handler,
                                                                                 dxl_id=self._dxl_id)
        if dxl_comm_result != COMM_SUCCESS:
            return False
        elif dxl_error != 0:
            return True
        else:
            return False

    def enable_dxl_torque(self) -> bool:
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=self._dxl_id,
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

    def disable_dxl_torque(self) -> bool:
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=self._dxl_id,
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

    def set_dxl_pos(self, tgt_pos: int = 0) -> bool:
        assert isinstance(tgt_pos, int)
        assert self._control_table.DXL_MINIMUM_POSITION_VALUE <= tgt_pos <= self._control_table.DXL_MAXIMUM_POSITION_VALUE
        dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(port=self._port_handler,
                                                                         dxl_id=self._dxl_id,
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

    def get_dxl_pos(self) -> int:
        dxl_present_position, dxl_comm_result, dxl_error = self._packet_handler.read4ByteTxRx(port=self._port_handler,
                                                                                              dxl_id=self._dxl_id,
                                                                                              address=self._control_table.ADDR_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            # print("%s" % self._packet_handler.getTxRxResult(dxl_comm_result))
            return -1
        elif dxl_error != 0:
            # print("%s" % self._packet_handler.getRxPacketError(dxl_error))
            return -1
        else:
            return dxl_present_position

    def enable_led(self) -> bool:
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=self._dxl_id,
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

    def disable_led(self) -> bool:
        dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(port=self._port_handler, dxl_id=self._dxl_id,
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


if __name__ == "__main__":
    from drivers.xarm.wrapper import XArmAPI

    ip = '192.168.1.232'
    peripheral_baud = 115200

    arm = XArmAPI(ip)
    time.sleep(2)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    dxl_con = XArmLite6DXLCon(arm, baudrate=peripheral_baud)
    dxl_con.enable_dxl_torque()
    dxl_con.set_dxl_pos(0)
    time.sleep(2)
    dxl_con.set_dxl_pos(4095)
    time.sleep(2)
    print(dxl_con.get_dxl_pos())
    dxl_con.disable_dxl_torque()
