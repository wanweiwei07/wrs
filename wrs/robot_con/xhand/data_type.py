import time
import struct
import serial
import numpy as np

# **CRC-16 Table (Same as `crc16tab` in C)**
CRC16_TABLE = [
    0x0000,0x1021,0x2042,0x3063,0x4084,0x50a5,0x60c6,0x70e7,
    0x8108,0x9129,0xa14a,0xb16b,0xc18c,0xd1ad,0xe1ce,0xf1ef,
    0x1231,0x0210,0x3273,0x2252,0x52b5,0x4294,0x72f7,0x62d6,
    0x9339,0x8318,0xb37b,0xa35a,0xd3bd,0xc39c,0xf3ff,0xe3de,
    0x2462,0x3443,0x0420,0x1401,0x64e6,0x74c7,0x44a4,0x5485,
    0xa56a,0xb54b,0x8528,0x9509,0xe5ee,0xf5cf,0xc5ac,0xd58d,
    0x3653,0x2672,0x1611,0x0630,0x76d7,0x66f6,0x5695,0x46b4,
    0xb75b,0xa77a,0x9719,0x8738,0xf7df,0xe7fe,0xd79d,0xc7bc,
    0x48c4,0x58e5,0x6886,0x78a7,0x0840,0x1861,0x2802,0x3823,
    0xc9cc,0xd9ed,0xe98e,0xf9af,0x8948,0x9969,0xa90a,0xb92b,
    0x5af5,0x4ad4,0x7ab7,0x6a96,0x1a71,0x0a50,0x3a33,0x2a12,
    0xdbfd,0xcbdc,0xfbbf,0xeb9e,0x9b79,0x8b58,0xbb3b,0xab1a,
    0x6ca6,0x7c87,0x4ce4,0x5cc5,0x2c22,0x3c03,0x0c60,0x1c41,
    0xedae,0xfd8f,0xcdec,0xddcd,0xad2a,0xbd0b,0x8d68,0x9d49,
    0x7e97,0x6eb6,0x5ed5,0x4ef4,0x3e13,0x2e32,0x1e51,0x0e70,
    0xff9f,0xefbe,0xdfdd,0xcffc,0xbf1b,0xaf3a,0x9f59,0x8f78,
    0x9188,0x81a9,0xb1ca,0xa1eb,0xd10c,0xc12d,0xf14e,0xe16f,
    0x1080,0x00a1,0x30c2,0x20e3,0x5004,0x4025,0x7046,0x6067,
    0x83b9,0x9398,0xa3fb,0xb3da,0xc33d,0xd31c,0xe37f,0xf35e,
    0x02b1,0x1290,0x22f3,0x32d2,0x4235,0x5214,0x6277,0x7256,
    0xb5ea,0xa5cb,0x95a8,0x8589,0xf56e,0xe54f,0xd52c,0xc50d,
    0x34e2,0x24c3,0x14a0,0x0481,0x7466,0x6447,0x5424,0x4405,
    0xa7db,0xb7fa,0x8799,0x97b8,0xe75f,0xf77e,0xc71d,0xd73c,
    0x26d3,0x36f2,0x0691,0x16b0,0x6657,0x7676,0x4615,0x5634,
    0xd94c,0xc96d,0xf90e,0xe92f,0x99c8,0x89e9,0xb98a,0xa9ab,
    0x5844,0x4865,0x7806,0x6827,0x18c0,0x08e1,0x3882,0x28a3,
    0xcb7d,0xdb5c,0xeb3f,0xfb1e,0x8bf9,0x9bd8,0xabbb,0xbb9a,
    0x4a75,0x5a54,0x6a37,0x7a16,0x0af1,0x1ad0,0x2ab3,0x3a92,
    0xfd2e,0xed0f,0xdd6c,0xcd4d,0xbdaa,0xad8b,0x9de8,0x8dc9,
    0x7c26,0x6c07,0x5c64,0x4c45,0x3ca2,0x2c83,0x1ce0,0x0cc1,
    0xef1f,0xff3e,0xcf5d,0xdf7c,0xaf9b,0xbfba,0x8fd9,0x9ff8,
    0x6e17,0x7e36,0x4e55,0x5e74,0x2e93,0x3eb2,0x0ed1,0x1ef0
]

FINGER_COMMAND_FORMAT = "<Hhhh f H H H H H H"  # LSB, 24 bytes
FINGER_STATE_FORMAT = "<BB f H H H H H H H H H"  # LSB, 22 bytes
SENSOR_DATA_FORMAT = "<bbb" + "bbb" * 120 + "B" * 20 + "B"  # 366 bytes (3, 120x3, 20, 1))

class FingerState:
    def __init__(self, id, sensor_id, position, torque, raw_position, temperature,
                 commboard_err, jointboard_err, tipboard_err, default5, default6, default7):
        self.id = id
        self.sensor_id = sensor_id
        self.position = position
        self.torque = torque
        self.raw_position = raw_position
        self.temperature = temperature
        self.commboard_err = commboard_err
        self.jointboard_err = jointboard_err
        self.tipboard_err = tipboard_err
        self.default5 = default5
        self.default6 = default6
        self.default7 = default7

    @classmethod
    def from_bytes(cls, data):
        unpacked = struct.unpack(FINGER_STATE_FORMAT, data)
        return cls(*unpacked)

class SensorData:
    def __init__(self, fx, fy, fz, force_data, temp_data, temp_total):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        self.force_data = force_data  # 120×(fx, fy, fz)
        self.temp_data = temp_data  # 20 temperature data
        self.temp_total = temp_total  # 1 temp sum

    @classmethod
    def from_bytes(cls, data):
        unpacked = struct.unpack(SENSOR_DATA_FORMAT, data)
        force_data = np.array([list(unpacked[3 + i * 3:3 + (i + 1) * 3]) for i in range(120)])
        temp_data = np.array(list(unpacked[3 + 120 * 3:3 + 120 * 3 + 20]))
        temp_total = unpacked[-1]
        return cls(unpacked[0], unpacked[1], unpacked[2], force_data, temp_data, temp_total)

class FingerCommand:
    def __init__(self, id=0, kp=0, ki=0, kd=0, position=0.0, tor_max=0, mode=0, res0=0, res1=0, res2=0, res3=0):
        self.id = id
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.position = position
        self.tor_max = tor_max
        self.mode = mode
        self.res0 = res0
        self.res1 = res1
        self.res2 = res2
        self.res3 = res3

    def to_bytes(self):
        return struct.pack(FINGER_COMMAND_FORMAT,
                           self.id, self.kp, self.ki, self.kd, self.position,
                           self.tor_max, self.mode, self.res0, self.res1, self.res2, self.res3)


class FingerCommandPackage:
    def __init__(self):
        self.commands = [FingerCommand() for _ in range(12)]

    def set_command(self, index, command):
        if 0 <= index < 12:
            self.commands[index] = command
        else:
            raise IndexError("Out of range (0~11)")

    def to_bytes(self):
        return b"".join(cmd.to_bytes() for cmd in self.commands)

def crc16(data):
    """ Compute CRC-16 using the same algorithm as the C implementation. """
    crc = 0x0000  # Initial value
    for byte in data:
        table_index = ((crc >> 8) ^ byte) & 0xFF
        crc = ((crc << 8) & 0xFFFF) ^ CRC16_TABLE[table_index]
    return struct.pack("<H", crc)  # Return little-endian CRC


def create_full_packet(command, data):
    """
    Create a full packet for RS-485 communication
    :param command: uint8_t 0x00~0xFF
    :param data: bytes
    :return:
    author: weiwei
    date: 20250218
    """
    frame_header = struct.pack("<H", 0x55AA)  # Head (uint16_t)
    src_id = struct.pack("<B", 0xFE)  # PC ID (uint8_t)
    dest_id = struct.pack("<B", 0x80)  # Hand ID (uint8_t)
    command = struct.pack("<B", command)  # Command (uint8_t)
    data_length = struct.pack("<H", len(data))  # data length (uint16_t)
    crc = crc16(frame_header + src_id + dest_id + command + data_length + data)
    return frame_header + src_id + dest_id + command + data_length + data + crc

def parse_rs485_response(response):
    # ignore if response is too short
    if len(response) < 7:
        print("Invalid data")
        return None
    # check header
    frame_header, src_id, dest_id, command, data_length = struct.unpack("<HBHBH", response[:7])
    if frame_header != 0x55AA:
        print("Wrong frame header")
        return None
    # check crc
    data_segment = response[7:-2]  # exclude frame header and CRC
    received_crc = struct.unpack("<H", response[-2:])[0]
    computed_crc = struct.unpack("<H", crc16(response[:-2]))[0]
    if received_crc != computed_crc:
        print("CRC check failed")
        return None
    # finger states
    finger_states = []
    offset = 0
    for i in range(12):
        finger_state_data = data_segment[offset:offset + 22]
        finger_states.append(FingerState.from_bytes(finger_state_data))
        offset += 22
    # sensor data
    sensor_data_list = []
    for i in range(5):
        sensor_data_bytes = data_segment[offset:offset + 366]
        sensor_data_list.append(SensorData.from_bytes(sensor_data_bytes))
        offset += 366
    return {
        "source_id": src_id,
        "dest_id": dest_id,
        "command": command,
        "finger_states": finger_states,
        "sensor_data": sensor_data_list
    }

def send_and_receive_rs485(port, baudrate, command_packet):
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1)
        if ser.is_open:
            print(f"send data: {command_packet.hex()}")
            ser.write(command_packet)
            ser.flush()
            time.sleep(0.002)
            response_size = 7 + 12 * 22 + 5 * 366 + 2
            response = ser.read(response_size)
            ser.close()
            return parse_rs485_response(response)
    except Exception as e:
        print(f"RS-485 发送/接收错误: {e}")
        return None

if __name__ == '__main__':
    finger_package = FingerCommandPackage()
    for i in range(12):
        cmd = FingerCommand(
            id=0x00 + i,
            kp=100,
            ki=50,
            kd=10,
            position=0,
            tor_max=300,
            mode=3,
            res0=0,
            res1=0,
            res2=0,
            res3=0)
        finger_package.set_command(i, cmd)
    full_packet = create_full_packet(0x80, finger_package.to_bytes())
    response_data = send_and_receive_rs485("COM3", 3000000, full_packet)
    # if response_data:
    #     print("成功接收回复:")
    #     print("源 ID:", response_data["source_id"])
    #     print("目的 ID:", response_data["dest_id"])
    #     print("命令字:", response_data["command"])
    #     print("手指状态:")
    #     for i, state in enumerate(response_data["finger_states"]):
    #         print(f"  手指 {i}: 位置={state.position:.2f}, 力矩={state.torque}, 温度={state.temperature}")
    #
    #     print("传感器数据:")
    #     for i, sensor in enumerate(response_data["sensor_data"]):
    #         print(f"  传感器 {i}: Fx={sensor.fx}, Fy={sensor.fy}, Fz={sensor.fz}, 温度合值={sensor.temp_total}")
    #         print(f"  120 组力数据 示例: {sensor.force_data[:5]}")
    #         print(f"  20 组温度数据 示例: {sensor.temp_data[:5]}")
    request_version = create_full_packet(0x13,  b"")

