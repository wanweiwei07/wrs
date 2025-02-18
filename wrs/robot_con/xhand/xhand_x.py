import time
import serial
import struct
import atexit
import wrs.robot_con.xhand.data_type as xhand_bt

class XHandX:
    def __init__(self, port="COM3", baudrate=3000000):
        """Initialize the serial connection to the XHandX device."""
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5)
            print(f"Connected to {port} at {baudrate} baud")
            # Register close function to be called at exit
            atexit.register(self.close)
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.ser = None  # Mark connection as failed

    def close(self):
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

    def calculate_crc(self, data):
        return xhand_bt.crc16(data)

    def send_command(self, command, data=b""):
        """Send a generic command with a data payload."""
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port is not open.")
            return None
        # Construct packet
        frame_header = struct.pack("<H", 0xAA55)  # Frame header (uint16_t)
        src_id = struct.pack("<B", 0xFE)  # Source ID (PC)
        dest_id = struct.pack("<B", 0x80)  # Destination ID (Hand)
        command = struct.pack("<B", command)  # Command (uint8_t)
        data_length = struct.pack("<H", len(data))  # Data length (uint16_t)
        crc = self.calculate_crc(frame_header + src_id + dest_id + command + data_length + data)
        packet = frame_header + src_id + dest_id + command + data_length + data + crc
        # Send command
        print(f"Sending: {packet.hex()}")
        self.ser.write(packet)
        self.ser.flush()
        time.sleep(0.002)
        # Wait for response
        return self.read_response()

    def read_response(self):
        """Read response from the serial device."""
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port is not open.")
            return None
        # Read header (7 bytes: frame header, IDs, command, length)
        response_header = self.ser.read(7)
        print(f"Received: {response_header.hex()}")
        if len(response_header) < 7:
            print("Error: Incomplete response header")
            return None
        # Extract data length
        _, _, _, command, data_length = struct.unpack("<HBBBH", response_header)
        data_length = int(data_length)
        # Read data and CRC (data_length + 2 bytes for CRC)
        response_data = self.ser.read(data_length + 2)
        if len(response_data) < data_length + 2:
            print("Error: Incomplete response data")
            return None
        data = response_data[:-2]  # Actual data
        received_crc = response_data[-2:]
        # Validate CRC
        computed_crc = self.calculate_crc(response_header + data)
        if received_crc != computed_crc:
            print("Error: CRC mismatch")
            return None
        print(f"Received: {response_header.hex()} {response_data.hex()}")
        return data

    def get_version(self):
        """Get firmware version."""
        return self.send_command(0x13)

    def goto_given_conf(self, jnt_values):
        """Move all 12 fingers to given joint configurations."""
        if len(jnt_values) != 12:
            raise ValueError("Expected exactly 12 joint values.")
        # **Create FingerCommandPackage**
        finger_package = xhand_bt.FingerCommandPackage()
        for i in range(12):
            cmd = xhand_bt.FingerCommand(
                id=i, kp=100, ki=0, kd=10,  # Default PID values
                position=jnt_values[i],  # Set position
                tor_max=300, mode=3, res0=0, res1=0, res2=0, res3=0
            )
            finger_package.set_command(i, cmd)
        # **Convert to bytes and send command**
        return self.send_command(0x02, finger_package.to_bytes())  # 0x02 = Move command

# **Example Usage**
if __name__ == "__main__":
    hand = XHandX(port="COM3", baudrate=3000000)
    hand.get_version()
    hand.goto_given_conf([0.5]*12)
    # Close connection
    hand.close()
