# -*- coding: utf-8 -*-

# Pysca: Yet another Visca, this time purely Python, implementation

# Copyright 2014 Rubén Pérez Vázquez

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import serial
import sys
import os
import select
import threading
import queue

# Timeout for read and write, in seconds
# TODO: More timeouts?
PORT_TIMEOUT = 5
RESPONSE_TIMEOUT = 30

# This is the default "offset" for the 'set address' command.
# According to the H-100 documentation, it is always one.
# libvisca makes this an argument, so we are including it here too
DEFAULT_SET_ADDR_OFFSET = 1

VISCA_BAUD_RATE = 9600
VISCA_BYTESIZE = 8
VISCA_STOPBITS = 1
VISCA_MIN_PKG_LEN = 3
VISCA_MAX_PKG_LEN = 16
VISCA_MAX_DEVICES = 7
VISCA_MAX_SOCKETS = 2
VISCA_TERMINATOR = 0xFF
VISCA_TERMINATOR_CHR = chr(VISCA_TERMINATOR)
VISCA_BCAST_ADDR = 0x08
VISCA_BCAST_MASK = 0b1000
VISCA_BCAST_HEADER = 0x88
VISCA_ADDR_MASK = 0b0111
VISCA_SENDER_MASK = 0b1000
VISCA_SENDER_MASK = 0x80
VISCA_VALID_ADDRS = range(0, VISCA_MAX_DEVICES + 1)

VISCA_HEADER_INDEX = 0
VISCA_TYPE_INDEX = 1
VISCA_CATEGORY_INDEX = 2

VISCA_COMMAND = 0x01
VISCA_INQUIRY = 0x09

VISCA_CATEGORY_INTERFACE = 0x00
VISCA_CATEGORY_CAMERA = 0x04
VISCA_CATEGORY_PAN_TILTER = 0x06
VISCA_CATEGORY_DISPLAY = 0x7e

VISCA_ADDR = 0x30
VISCA_ADDR_SET = 0x30
VISCA_ADDR_CHANGE = 0x38

VISCA_RESPONSE_ACK = 0x40
VISCA_RESPONSE_COMPLETED = 0x50
VISCA_RESPONSE_ERROR = 0x60
VISCA_RESPONSES = [VISCA_ADDR, VISCA_RESPONSE_ACK, \
                   VISCA_RESPONSE_COMPLETED, VISCA_RESPONSE_ERROR]

VISCA_ERROR_MESSAGE_LENGTH = 0x01
VISCA_ERROR_SYNTAX = 0x02
VISCA_ERROR_CMD_BUFFER_FULL = 0x03
VISCA_ERROR_CMD_CANCELLED = 0x04
VISCA_ERROR_NO_SOCKET = 0x05
VISCA_ERROR_CMD_NOT_EXECUTABLE = 0x41

VISCA_IF_CLEAR = 0x01
VISCA_IF_CLEAR_PAYLOAD = serial.to_bytes([VISCA_COMMAND, \
                                          VISCA_CATEGORY_INTERFACE, \
                                          VISCA_IF_CLEAR])
VISCA_POWER = 0x00
VISCA_POWER_ON = 0x02
VISCA_POWER_OFF = 0x03

VISCA_ZOOM = 0x07
VISCA_ZOOM_STOP = 0x00
VISCA_ZOOM_TELE = 0x02
VISCA_ZOOM_WIDE = 0x03
VISCA_ZOOM_TELE_SPEED = 0x20
VISCA_ZOOM_WIDE_SPEED = 0x30
VISCA_ZOOM_VALUE = 0x47

ZOOM_ACTION_STOP = 'stop'
ZOOM_ACTION_TELE = 'tele'
ZOOM_ACTION_WIDE = 'wide'

VISCA_DZOOM = 0x06
VISCA_DZOOM_ON = 0x02
VISCA_DZOOM_OFF = 0x03

VISCA_FOCUS = 0x08
VISCA_FOCUS_STOP = 0x00
VISCA_FOCUS_FAR = 0x02
VISCA_FOCUS_NEAR = 0x03
VISCA_FOCUS_FAR_VALUE = 0x20
VISCA_FOCUS_NEAR_VALUE = 0x30
VISCA_FOCUS_VALUE = 0x48

FOCUS_ACTION_STOP = 'stop'
FOCUS_ACTION_FAR = 'far'
FOCUS_ACTION_NEAR = 'near'

VISCA_FOCUS_AUTO = 0x38
VISCA_FOCUS_AUTO_ON = 0x02
VISCA_FOCUS_AUTO_OFF = 0x03
VISCA_FOCUS_AUTO_SWITCH = 0x10

VISCA_FOCUS_TRIGGER = 0x18
VISCA_FOCUS_TRIGGER_TRIGGER = 0x01
VISCA_FOCUS_TRIGGER_INFINITY = 0x02

VISCA_FOCUS_NEAR_LIMIT = 0x28

FOCUS_AUTO_MODE_MANUAL = 'manual'
FOCUS_AUTO_MODE_AUTO = 'auto'
FOCUS_TRIGGER_MODE_INFINITY = 'infinity'
FOCUS_TRIGGER_MODE_TRIGGER = 'trigger'

VISCA_FOCUS_AUTO_SENSE = 0x58
VISCA_FOCUS_AUTO_SENSE_HIGH = 0x02
VISCA_FOCUS_AUTO_SENSE_LOW = 0x03

VISCA_FOCUS_AUTO_MOV = 0x57
VISCA_FOCUS_AUTO_MOV_NORMAL = 0x00
VISCA_FOCUS_AUTO_MOV_INTERVAL = 0x01
VISCA_FOCUS_AUTO_MOV_ZOOM = 0x02

FOCUS_AUTO_MOV_MODE_NORMAL = 'normal'
FOCUS_AUTO_MOV_MODE_INTERVAL = 'interval'
FOCUS_AUTO_MOV_MODE_ZOOM = 'zoom'

VISCA_FOCUS_AUTO_ACTIVE_INTERVAL = 0x27

VISCA_IR_CORRECTION = 0x11
VISCA_IR_CORRECTION_ON = 0x00
VISCA_IR_CORRECTION_OFF = 0x01

VISCA_WB = 0x35
VISCA_WB_AUTO = 0x00
VISCA_WB_INDOOR = 0x01
VISCA_WB_OUTDOOR = 0x02
VISCA_WB_ONEPUSH = 0x03
VISCA_WB_MANUAL = 0x05
VISCA_WB_TRIGGER = 0x10
VISCA_WB_TRIGGER_ONEPUSH = 0x05

WB_AUTO_MODE = 'auto'
WB_INDOOR_MODE = 'indoor'
WB_OUTDOOR_MODE = 'outdoor'
WB_ONEPUSH_MODE = 'onepush'
WB_MANUAL_MODE = 'manual'

VISCA_RGAIN = 0x03
VISCA_RGAIN_RESET = 0x00
VISCA_RGAIN_UP = 0x02
VISCA_RGAIN_DOWN = 0x03
VISCA_RGAIN_VALUE = 0x43

VISCA_BGAIN = 0x04
VISCA_BGAIN_RESET = 0x00
VISCA_BGAIN_UP = 0x02
VISCA_BGAIN_DOWN = 0x03
VISCA_BGAIN_VALUE = 0x44

VISCA_AUTO_EXPOSURE = 0x39
VISCA_AUTO_EXPOSURE_FULL_AUTO = 0x00
VISCA_AUTO_EXPOSURE_MANUAL = 0x03
VISCA_AUTO_EXPOSURE_SHUTTER_PRIORITY = 0x0A
VISCA_AUTO_EXPOSURE_IRIS_PRIORITY = 0x0B
VISCA_AUTO_EXPOSURE_BRIGHT = 0x0D

AUTO_EXPOSURE_FULL_AUTO_MODE = 'auto'
AUTO_EXPOSURE_MANUAL_MODE = 'manual'
AUTO_EXPOSURE_SHUTTER_PRIORITY_MODE = 'shutter'
AUTO_EXPOSURE_IRIS_PRIORITY_MODE = 'iris'
AUTO_EXPOSURE_BRIGHT_MODE = 'bright'

VISCA_BRIGHT = 0x0D
VISCA_BRIGHT_UP = 0x02
VISCA_BRIGHT_DOWN = 0x03
VISCA_BRIGHT_VALUE = 0x4D

BRIGHT_ACTION_UP = 'up'
BRIGHT_ACTION_DOWN = 'down'

VISCA_EXPOSURE_COMP = 0x3E
VISCA_EXPOSURE_COMP_ON = 0x02
VISCA_EXPOSURE_COMP_OFF = 0x03
VISCA_EXPOSURE_COMP_AMOUNT = 0x0E
VISCA_EXPOSURE_COMP_RESET = 0x00
VISCA_EXPOSURE_COMP_UP = 0x02
VISCA_EXPOSURE_COMP_DOWN = 0x03
VISCA_EXPOSURE_COMP_VALUE = 0x4E

EXPOSURE_COMP_ACTION_ON = 'on'
EXPOSURE_COMP_ACTION_OFF = 'off'
EXPOSURE_COMP_ACTION_RESET = 'reset'
EXPOSURE_COMP_ACTION_UP = 'up'
EXPOSURE_COMP_ACTION_DOWN = 'down'

VISCA_MEMORY = 0x3F
VISCA_MEMORY_RESET = 0x00
VISCA_MEMORY_SET = 0x01
VISCA_MEMORY_RECALL = 0x02

MAX_MEMORY_POSITIONS = 6

VISCA_PT_DRIVE = 0x01
VISCA_PT_DRIVE_HORIZ_LEFT = 0x01
VISCA_PT_DRIVE_HORIZ_RIGHT = 0x02
VISCA_PT_DRIVE_HORIZ_STOP = 0x03
VISCA_PT_DRIVE_VERT_UP = 0x01
VISCA_PT_DRIVE_VERT_DOWN = 0x02
VISCA_PT_DRIVE_VERT_STOP = 0x03
VISCA_PT_ABSOLUTE_POSITION = 0x02
VISCA_PT_RELATIVE_POSITION = 0x03
VISCA_PT_HOME = 0x04
VISCA_PT_RESET = 0x05
VISCA_PT_LIMITSET = 0x07
VISCA_PT_LIMITSET_SET = 0x00
VISCA_PT_LIMITSET_CLEAR = 0x01
VISCA_PT_LIMITSET_SET_UR = 0x01
VISCA_PT_LIMITSET_SET_DL = 0x00

VISCA_INFO_DISPLAY = 0x18
VISCA_INFO_DISPLAY_ON = 0x02
VISCA_INFO_DISPLAY_OFF = 0x03

H_NIBBLE_MASK = 0xF0
L_NIBBLE_MASK = 0x0F


# EXCEPTIONS
class ViscaError(RuntimeError):
    pass


class ViscaTimeoutError(ViscaError):
    pass


class ViscaSocketStatusError(ViscaError):
    pass


class ViscaNoSuchDeviceError(ViscaError):
    pass


class ViscaDeviceNotReadyError(ViscaError):
    pass


class ViscaResponseError(ViscaError):
    pass


class ViscaMessageLengthError(ViscaError):
    pass


class ViscaSyntaxError(ViscaResponseError):
    pass


class ViscaBufferFullError(ViscaResponseError):
    pass


class ViscaCommandCancelledError(ViscaResponseError):
    pass


class ViscaNoSocketError(ViscaResponseError):
    pass


class ViscaCommandNotExecutableError(ViscaResponseError):
    pass


class ViscaUnexpectedResponseError(ViscaResponseError):
    pass


class Packet(bytes):
    """
    This class models a packet of data, as specified by the Visca protocol.
    It implements some safety checks, and includes also some static utility functions.
    """

    # Map the known errors returned by the Visca devices to the appropriate exceptions
    __error_map = {VISCA_ERROR_MESSAGE_LENGTH: lambda x: ViscaMessageLengthError(x),
                   VISCA_ERROR_SYNTAX: lambda x: ViscaSyntaxError(x),
                   VISCA_ERROR_CMD_BUFFER_FULL: lambda x: ViscaBufferFullError(x),
                   VISCA_ERROR_CMD_CANCELLED: lambda x: ViscaCommandCancelledError(x),
                   VISCA_ERROR_NO_SOCKET: lambda x: ViscaNoSocketError(x),
                   VISCA_ERROR_CMD_NOT_EXECUTABLE: lambda x: ViscaCommandNotExecutableError(x)}

    def __init__(self, packet=None):
        # Check argument
        if packet is None:
            packet = ""

        # Initialize this
        super(Packet, self).__init__(packet)

        if len(self) < VISCA_MIN_PKG_LEN or \
                len(self) > VISCA_MAX_PKG_LEN:
            raise ValueError("Incorrect packet size for '{0}': {1}".format(self, len(self)))

        if ord(self[-1]) != VISCA_TERMINATOR:
            raise ValueError("Incorrect terminator byte")

        # TODO: Check packet syntax???

    def __str__(self):
        return self.encode('hex')

    @classmethod
    def from_serial(cls, port):
        """
        Build a new packet with bytes read from a 'serial' object.
        This method does not block on read. I.e., if the packet cannot
        be read from the serial port without blocking, then an exception
        is raised.
        """
        if not isinstance(port, serial.Serial):
            raise TypeError("The argument must be an instance of the 'Serial' class")

        # Empty 'bytes' instance
        packet = serial.to_bytes([])

        for count in xrange(VISCA_MAX_PKG_LEN):
            # Read a new byte
            new_byte = port.read(1)

            # If it is empty, read has returned for timeout
            if len(new_byte) == 0:
                break

            # Otherwise, append the read byte to the packet contents
            packet = packet + new_byte

            # Check whether we received a terminator
            if ord(new_byte) == VISCA_TERMINATOR:
                break

        # The error handling (packet size, packet terminator, etc.) is done on the constructor
        return cls(packet)

    @classmethod
    def from_parts(cls, sender=None, recipient=None, *parts):
        """
        Generate a packet from their parts.
        The "parts" here are the sender and the recipient IDs (both are integers from 1 to 8), plus
        any number of arguments that should be directly convertible into the "byte" end_type, or iterable
        objects containing objects of that end_type.
        """

        if sender is None:
            sender = 0
        if recipient is None:
            recipient = VISCA_BCAST_ADDR

        payload = bytes()
        for part in parts:
            # This is to check if "part" is iterable, because the function serial.to_bytes
            # expects an iterable object
            try:
                iter(part)
                p = part
            except TypeError:
                # 'part' is not iterable
                p = [part]

            payload = payload + serial.to_bytes(p)

        if payload and ord(payload[-1]) == VISCA_TERMINATOR:
            payload = payload[:-1]

        return cls(Packet.header_for(sender, recipient) + payload + serial.to_bytes([VISCA_TERMINATOR]))

    @staticmethod
    def header_for(sender, recipient):
        """
        Calculate a Packet's header from its components, the sender and recipiend IDs
        """
        if sender in VISCA_VALID_ADDRS:
            if recipient in VISCA_VALID_ADDRS or recipient == VISCA_BCAST_ADDR:
                return serial.to_bytes([(sender << 4) | VISCA_SENDER_MASK | recipient])
            else:
                raise ValueError("Invalid recipient: {}".format(recipient))
        else:
            ValueError("Invalid sender: {}".format(sender))

    @staticmethod
    def int_to_bytes(number, size=None):
        """
        Convert an integer into the nibble-separated byte sequence used by Visca.
        For instance, the integer 0xpqrs would become 0x0p0q0r0s, where p, q, r and s are
        hexadecimal numbers from 0 to F.
        If the first parameter is not an integer or cannot be converted to one, ValueError is raised.
        The second parameter is an optional "size" indicator, that tells the number of
        bytes that the number should be represented with. If not specified, this method
        returns just the necessary number of bytes to represent the number in the
        aforementioned way.
        If the number requires more that 'size' bytes to be represented in the way
        described above, ValueError is raised.
        """
        try:
            n = int(number)
        except ValueError as e:
            raise ValueError("'{}' cannot be converted to an integer".format(number), e)

        if n < 0:
            raise ValueError("'{}' is not a positive number".format(number))

        if size is None or size == 0:
            s = 0
        elif size > 0:
            # Make sure size is an integer
            s = int(size)
        else:
            raise ValueError("'{}' should be the axis_length of the encoded number".format(size))

        response = bytes()
        # Go on while number is not 0 or size is not 0
        while n or s:
            response = serial.to_bytes([n & L_NIBBLE_MASK]) + response
            n >>= 4
            if s:
                s -= 1
                if s == 0 and n > 0:
                    raise ValueError(
                        "The argument '{}' is too high to fit in the specified size '{}'".format(number, size))

        return response

    def parse_error(self):
        """
        Raise the appropriate exception according to the error sent by the camera.
        If this packet does not represent an error, return the packet itself
        """

        if self.type == VISCA_RESPONSE_ERROR:
            try:
                raise Packet.__error_map[self.category](self)
            except KeyError:
                raise ViscaUnexpectedResponseError(self)
        return self

    @property
    def header(self):
        return ord(self[VISCA_HEADER_INDEX])

    @property
    def payload(self):
        return self[1:-1]

    @property
    def sender(self):
        return (self.header >> 4) & VISCA_ADDR_MASK

    @property
    def recipient(self):
        if self.header & VISCA_BCAST_MASK:
            return VISCA_BCAST_ADDR
        else:
            return self.header & VISCA_ADDR_MASK

    @property
    def type(self):
        return ord(self[VISCA_TYPE_INDEX]) & H_NIBBLE_MASK

    @property
    def socket(self):
        return ord(self[VISCA_TYPE_INDEX]) & L_NIBBLE_MASK

    @property
    def subtype(self):
        return ord(self[VISCA_TYPE_INDEX]) & L_NIBBLE_MASK

    @property
    def category(self):
        if ord(self[VISCA_CATEGORY_INDEX]) != VISCA_TERMINATOR:
            return ord(self[VISCA_CATEGORY_INDEX])
        else:
            return None


class Socket(object):
    # Socket status codes
    READY = 0
    WAITING = 1
    CLEARING = 2

    def __init__(self, number=0, timeout=None):
        super(Socket, self).__init__()
        self.__number = number
        self.__cond = threading.Condition()
        self.packet_sent = None
        self.packet_received = None
        self.__status = Socket.READY
        self.__timeout = timeout
        self.__waiting = 0

    def get_response(self, timeout=None):
        # Make sure there's only one new request at a time
        with self.__cond:
            if timeout is None:
                timeout = self.__timeout

            wait = True
            # This will loop at most once, if status is WAITING and wait is True (just the first time)
            while self.__status != Socket.READY:
                if self.__status == Socket.WAITING:
                    if wait:
                        self.__waiting += 1
                        self.__cond.wait(timeout)
                        self.__waiting -= 1
                        wait = False
                    else:
                        raise ViscaTimeoutError(
                            "Socket {} did not return an answer after {} seconds".format(self.__number, timeout))
                elif self.__status == Socket.CLEARING:
                    raise ViscaSocketStatusError("Socket {} is being cleared and can be no longer read" \
                                                 .format(self.__number))
                else:
                    # This shouldn't happen
                    raise ViscaSocketStatusError(
                        "Socket {} is in the unknown status {}".format(self.__number, self.__status))

            # Read the socket contents
            return self.packet_received

    def clear(self):
        with self.__cond:
            if self.__waiting > 0:
                if self.__status == Socket.READY:
                    print
                    "WARNING: Clearing socket {} found {} threads waiting but status was READY".format(self.__number,
                                                                                                       self.__waiting)

            # Awake all the waiting threads. After they all fail, return to READY state
            self.__status = Socket.CLEARING
            while self.__waiting > 0:
                # Awake all the threads waiting for reading
                self.__cond.notify_all()
                # TODO Timeout?
                self.__cond.wait()

            # Reset status to READY and reset the received packet
            self.__status = Socket.READY
            self.packet_sent = None
            self.packet_received = None

    def wait_for_response(self, packet_sent):
        with self.__cond:
            if self.__status == Socket.READY:
                self.__status = Socket.WAITING
                self.packet_sent = packet_sent
                self.packet_received = None
            elif self.__status == Socket.WAITING:
                raise ViscaSocketStatusError(
                    "Tried to block socket {} for packet '{}', but socket was already WAITING response for packet '{}'" \
                    .format(self.__number, packet_sent, self.packet_sent))
            else:
                raise ViscaSocketStatusError(
                    "Tried to block socket {} for packet '{}', but socket was in unexpected status {}" \
                    .format(self.__number, packet_sent, self.__status))

    def recv(self, packet):
        with self.__cond:
            # Check socket status
            if self.__status == Socket.WAITING:
                # print "Received packet '{}' in socket {}".format(packet, self.__number)
                # Make the reception of the packet on this socket
                self.packet_received = packet
                # Mark socket as READY
                self.__status = Socket.READY
                # Signal the packet reception
                self.__cond.notify_all()
            else:
                raise ViscaSocketStatusError("Socket {} received packet '{}', but was in unexpected status {}" \
                                             .format(self.__number, packet, self.__status))


class Device(object):

    def __init__(self, address, send_handler, sockets=VISCA_MAX_SOCKETS, timeout=None):
        # TODO: Check sockets exceeds maximum? Make it constant?
        self.address = address
        # Lock to ensure the new requests wait until we get a response from the camera
        self.__send_lock = threading.RLock()
        # Create reception sockets
        # Number 0 is for the normal operation.
        # Numbers 1 and above are for ack'ed commands awaiting for a "completed" response
        self.__sockets = [Socket(i, timeout) for i in range(sockets + 1)]
        # The handler used to send the requests to the device
        self.__send = send_handler

    def send(self, *payload, **kwargs):
        with self.__send_lock:
            # Send a command or request to the device
            packet = Packet.from_parts(0, self.address, *payload)
            self.__sockets[0].wait_for_response(packet)
            self.__send(packet)
            response = self.__sockets[0].get_response()
            if response.type == VISCA_RESPONSE_ACK and kwargs["blocking"]:
                return self.__sockets[response.socket].get_response()
            else:
                return response

    def get_response(self, socket=0):
        try:
            return self.__sockets[socket].get_response()
        except KeyError:
            raise ViscaNoSuchDeviceError("Tried to read response from a non-existant socket: {}".format(socket))

    def recv(self, packet):
        try:
            if packet.type == VISCA_RESPONSE_COMPLETED:
                self.__sockets[packet.socket].recv(packet)
            else:
                if packet.type == VISCA_RESPONSE_ACK:
                    # Mark the corresponding socket to wait for the "COMPLETED" packet
                    self.__sockets[packet.socket].wait_for_response(self.__sockets[0].packet_sent)

                # Signal the reception in the "main" socket (socket 0)
                self.__sockets[0].recv(packet)
        except KeyError:
            raise ViscaNoSuchDeviceError(
                "Tried to register a response in a non-existant socket: {}".format(packet.socket))

    def clear(self):
        # This is why we used a reentrant lock
        # When clearing the device, we use this lock twice to avoid new requests to go on while the "sockets" are being cleared
        with self.__send_lock:
            # Send the "clear" packet
            response = self.send(VISCA_IF_CLEAR_PAYLOAD)
            # Create one thread per socket
            threads = [threading.Thread(target=socket.clear()) for socket in self.__sockets]
            # Run the threads in parallel
            for t in threads:
                t.start()
            # Wait for the threads to finish
            for t in threads:
                t.join()


__serialport = None
__timeout = None
__port_available = threading.Lock()
__portname = None
__devices = {}
__alive = False
__reader_thread = None
__bcast_reader_thread = None

# For "special" packets
__init_addresses_lock = threading.Condition()
__init_addresses_rcvd = False
__init_addresses_offset = 1
__if_clear_lock = threading.Condition()
__if_clear_rcvd = False

__bcast_queue = queue.Queue()

# TODO: Is this needed?
# self._allow_write = threading.Event()
# self._allow_write.clear()

# used for signaling
__pipe = os.pipe()


def __reader():
    # TODO Handle the "close" command gracefully
    while (__alive):
        try:
            # Block till there are bytes to read
            # self.pipe is a method for releasing the block on shutdown
            # print "__READER SELECT. THERE ARE", self.__serialport.inWaiting(), "PENDING BYTES"
            ready, _, _ = select.select([__pipe[0], __serialport], [], [])

            # Read the packet
            # print "__READER READ PACKET"
            p = Packet.from_serial(__serialport)
            # print "Received packet:", p

            # Check the response according to several types
            if p.header == VISCA_BCAST_HEADER:
                __bcast_queue.put(p)
            elif ord(p[1]) == VISCA_ADDR_CHANGE:
                # print "__READER: Received network change packet!!!"
                # Read responses in a new thread
                threading.Thread(target=cmd_address_set()).start()
            else:
                # Make the reception of the packet in the corresponding device
                # Responses with a sender of "0" are broadcast requests, but are duly handled by the device 0
                # print "Making reception of packet in device", p.sender
                # TODO: Refactor this
                __devices[p.sender].recv(p)
                # TODO: Use logger
        except KeyError:
            # TODO: Do not raise exception, or handle gracefully
            # TODO: Use logger
            print
            "WARN: Received packet from an unregistered sender {}: {}".format(p.sender, p)
            # raise ViscaNoSuchDeviceError("Received packet from an unregistered sender {}: {}".format(p.sender, p))
        except ValueError as e:
            # TODO: Do not raise exception, or handle gracefully
            # TODO: Use logger
            print
            "WARN: Received an incorrect packet: {}".format(e)
            # raise ValueError("Received an incorrect packet", e)


def __bcast_reader():
    global __if_clear_rcvd
    global __devices

    while (__alive):
        # Read from the bcast queue
        packet = __bcast_queue.get()

        if packet.payload == VISCA_IF_CLEAR_PAYLOAD:
            with __if_clear_lock:
                # Clear all devices
                __clear_all()
                # Signal the reception
                __if_clear_rcvd = True
                __if_clear_lock.notify()

        elif ord(packet[1]) == VISCA_ADDR_SET:
            with __init_addresses_lock:
                new_addr = ord(packet[2]) - __init_addresses_offset
                if new_addr not in __devices:
                    __devices[new_addr] = Device(new_addr, __write_to_serial, timeout=__timeout)
                else:
                    # TODO: Use logger
                    # TODO: Any other measures?
                    print
                    "WARNING: Received 'SET_ADDR' package for existing device {}".format(new_addr)
                # Signal the reception
                __init_addresses_rcvd = True
                __init_addresses_lock.notify()
        else:
            # TODO: Use logger
            print
            "WARNING: Broadcast reader received unknown packet: {}".format(packet)

        # Signal the task is done
        __bcast_queue.task_done()


def __write_to_serial(packet):
    # Block till the serial port is available
    with __port_available:
        try:
            __serialport.write(packet)
        except serial.SerialTimeoutException as e:
            # TODO User logger
            print
            "ERROR: Timeout received when writing to the serial port"
            raise ViscaTimeoutError(e)
        except serial.portNotOpenError as e:
            # TODO User logger
            print
            "ERROR: Port is not open when writing"
            # TODO: Gracefully signal all threads to stop
            raise ViscaError(e)


def __clear_all():
    # TODO: Exceptions?
    ths = [threading.Thread(target=__devices[i].clear) for i in __devices]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def connect(portname, timeout=None):
    """
    Initialize the library.
    Arguments:
       * portname: The path to the serial port to where the Visca device(s) are connected.
       * timeout: (Optional) An optional read timeout.
    """

    global __timeout
    global __serialport
    global __reader_thread
    global __bcast_reader_thread
    global __port_available
    global __alive

    # Initialize __timeout
    __timeout = timeout if timeout else RESPONSE_TIMEOUT

    with __port_available:
        if (__serialport == None):
            try:
                # Initialize port
                __serialport = serial.Serial(portname, \
                                             VISCA_BAUD_RATE, \
                                             timeout=PORT_TIMEOUT, \
                                             stopbits=VISCA_STOPBITS, \
                                             bytesize=VISCA_BYTESIZE)
                __serialport.flushInput()
            except Exception as e:
                # Close serial port
                try:
                    __serialport.close()
                except Exception:
                    # Ignored
                    pass
                finally:
                    __serialport = None

                raise ViscaError("Could not open serial port '%s' for display: %s\n" % (portname, e))

    # Init threads
    __alive = True
    __reader_thread = threading.Thread(target=__reader, name="reader-thread")
    __bcast_reader_thread = threading.Thread(target=__bcast_reader, name="bcast-reader-thread")
    __reader_thread.daemon = True
    __bcast_reader_thread.daemon = True
    __reader_thread.start()
    __bcast_reader_thread.start()

    # Initialize addresses
    init_addresses()


def init_addresses(first=DEFAULT_SET_ADDR_OFFSET):
    """
    Sends a request to all the devices to reconfigure their addresses.
    When a device is plugged or unplugged, Pysca sends this command automatically,
    so the users do not need to call it themselves.
    """
    global __init_addresses_offset
    global __init_addresses_rcvd
    global __devices

    with __init_addresses_lock:
        # Set the address offset (defaults to 1)
        try:
            __init_addresses_offset = int(first)
        except (TypeError, ValueError):
            # TODO User logger
            # Ignore the exception and set the default value
            print
            "WARNING: Ignore invalid first address value in 'init_addresses': {}".format(first)
            __init_addresses_offset = DEFAULT_SET_ADDR_OFFSET

        if __init_addresses_rcvd:
            # TODO User logger
            print
            "WARNING: The flag 'address set received' was already set when an 'address set' command was issued"
            __init_addresses_rcvd = False

        # Flush the pending commands
        __clear_all()

        # Send the address set request
        __write_to_serial(Packet.from_parts(0, VISCA_BCAST_ADDR, VISCA_ADDR, first))

        # Delete the existing devices
        __devices = {}

        # Wait for the responses to arrive.
        # The packets are handled by the broadcast receiving loop, but we keep track of the packets received
        # so that the method does not return until we wait for __timeout seconds for a new response.
        # We assume that if a new packet takes more than __timeout  seconds to arrive, then no more packets
        # will arrive, so we return
        # Even after that, should a new 'set address' packet be received, it would be processed anyway
        while not __init_addresses_rcvd:

            # Sleep till another packet is received, or the timeout occurs
            # TODO: Use longer timeout?
            __init_addresses_lock.wait(__timeout)

            # Check whether a timeout or a reception occured
            if __init_addresses_rcvd:
                # Reception
                # TODO: Use logger
                # print "timeout: {}".format(self.__timeout)
                # print "Received new address set packet for device {}. {}".format(self.__init_addresses_rcvd, self.__devices)
                # Reset flag
                __init_addresses_rcvd = False
            else:
                # Timeout. We assume no more 'set_address' packets are coming
                break


def clear_all():
    """
    Cleans the command buffers for all the connected devices
    """

    global __if_clear_rcvd

    with __if_clear_lock:
        # Check if the flag is true --it shouldn't
        if __if_clear_rcvd:
            # TODO Use logger
            print
            "WARNING: The flag 'if_clear received' was already set when an 'if_clear' command was issued"
            __if_clear_rcvd = False

        # Send command
        resp = __write_to_serial(Packet.from_parts(0, VISCA_BCAST_ADDR, VISCA_IF_CLEAR_PAYLOAD))

        # Wait for a response
        # TODO: Use longer timeout?
        __if_clear_lock.wait(__timeout)

        # Check if it was a timeout or the packet was received
        if __if_clear_rcvd:
            # The real operation is handled by the broadcast reception loop.
            # Here, simply signal the reception and return
            __if_clear_rcvd = False
        else:
            # Timeout
            raise ViscaTimeoutError("Timeout waiting for a response to an \"clear_all\" command")


def clear_commands(dest):
    """
    Clean the command buffers
    The destination may be the broadcast address (8), in which case, all the connected devices will be cleared
    """
    if dest == VISCA_BCAST_ADDR:
        clear_all()
    else:
        __devices[dest].send(VISCA_IF_CLEAR_PAYLOAD)
        try:
            __devices[i].clear()
        except KeyError:
            # Ignore. This command may be send on initialization, when
            # the different 'Device' instances are not yet created
            pass


def __cmd_cam(device, *parts, **kwargs):
    return __devices[device].send(VISCA_COMMAND, VISCA_CATEGORY_CAMERA, *parts, **kwargs).parse_error()


def __cmd_pt(device, *parts, **kwargs):
    return __devices[device].send(VISCA_COMMAND, VISCA_CATEGORY_PAN_TILTER, *parts, **kwargs).parse_error()


def __cmd_dis(device, *parts, **kwargs):
    return __devices[device].send(VISCA_COMMAND, VISCA_CATEGORY_DISPLAY, *parts, **kwargs).parse_error()


# POWER control
def set_power_on(device, on):
    """
    Powers the camera on and off.
    The action depends on the parameter "on".
    * If it is True, try to set the camera on.
    * If it is False, try to set the camera off.
    """
    if on:
        __cmd_cam(device, VISCA_POWER, VISCA_POWER_ON, blocking=True)
    else:
        __cmd_cam(device, VISCA_POWER, VISCA_POWER_OFF, blocking=True)


################
# ZOOM control #
################
def zoom(device, action, speed=None, focus=None, blocking=False):
    """
    Zoom the camera in or out.
    The second parameter may take any of the following forms:
       * 'tele' for starting a close-up movement
       * 'wide' for starting a zoom-out movement
       * 'stop' for stopping a movement started with the previous commands
       * A four-byte integer, to set up a fixed zoom position. Please find the possible values and their meaning
         on the device's manual.
    The optional third parameter 'speed':
       * In actions 'tele' and 'wide', specifies a zoom speed value ranging from 0 to 7
       * Has no effect in any other case.
    When the second parameter is a fixed zoom position, an optional 'focus' argument
    can be specified, indicating a fixed focus position to set the device to.
    """
    # Mechanism to convert the actions accepted by the 'zoom' command into Visca codes
    actions2codes = {ZOOM_ACTION_STOP: lambda speed: VISCA_ZOOM_STOP,
                     ZOOM_ACTION_TELE: lambda
                         speed: VISCA_ZOOM_TELE if speed is None else VISCA_ZOOM_TELE_SPEED | speed,
                     ZOOM_ACTION_WIDE: lambda
                         speed: VISCA_ZOOM_WIDE if speed is None else VISCA_ZOOM_WIDE_SPEED | speed}

    if action == ZOOM_ACTION_TELE or action == ZOOM_ACTION_WIDE:
        # Be flexible about the values that can be passed to 'speed',
        # but only when speed is really meaningful
        try:
            # Convert speed into an integer if possible
            speed = int(speed)

            # Make sure it is between the boundaries
            if speed < 0:
                speed = 0
            elif speed > 7:
                speed = 7
        except ValueError as e:
            # We did what we could...
            e.message = "The value of speed should be an integer between 0 and 7"
            raise

    try:
        __cmd_cam(device, VISCA_ZOOM, actions2codes[action](speed), blocking=blocking)
    except KeyError:
        # 'action' must not be a known action (i.e. is not a key in the 'actions2codes' dictionary), but an absolute zoom position
        # This will raise an exception if action cannot be converted to an integer
        if focus is None:
            __cmd_cam(device, VISCA_ZOOM_VALUE, Packet.int_to_bytes(action, 4), blocking=blocking)
        else:
            # Use the operation that sets up the focus and the zoom at the same time
            __cmd_cam(device, VISCA_FOCUS_VALUE, Packet.int_to_bytes(action, 4), Packet.int_to_bytes(focus, 4),
                      blocking=blocking)


def set_zoom(device, zoom, focus=None, blocking=False):
    """
    Sets the zoom to a fixed position.
    The 'zoom' argument is a 4-byte integer. The possible values and their meaning can be consulted in the
    device's manual.
    An optional 'focus' argument will set the device focus to the indicated position in one operation.
    'focus' should be a 4-byte integer, its acceptable values and their meaning to be found at the device's manual.
    """
    if focus is None:
        __cmd_cam(device, VISCA_ZOOM_VALUE, Packet.int_to_bytes(zoom, 4), blocking=blocking)
    else:
        # Sets focus and zoom at the same time
        __cmd_cam(device, VISCA_ZOOM_VALUE, Packet.int_to_bytes(zoom, 4), Packet.int_to_bytes(focus, 4),
                  blocking=blocking)


# Digital Zoom control on/off
def set_digital_zoom(device, on=True, blocking=False):
    """
    Set the digital zoom on and off.
    The action performed depends on the parameter "on".
       * If it is True, or unspecified, activate the digital zoom.
       * If it is False, deactivate the digital zoom.
    """
    if on:
        __cmd_cam(device, VISCA_DZOOM, VISCA_DZOOM_ON, blocking=blocking)
    else:
        __cmd_cam(device, VISCA_DZOOM, VISCA_DZOOM_OFF, blocking=blocking)


def focus(device, action, speed=None, zoom=None, blocking=False):
    """
    Changes camera focus.
    The second parameter may take any of the following forms:
       * 'far' starts moving the focus towards far objects
       * 'near' starts moving the focus towards near objects
       * 'stop' stops a movement started with the previous commands
    The optional third parameter 'speed':
       * In actions 'far' and 'near', specifies a focus speed value ranging from 0 to 7
       * Has no effect in any other case.
    When the second parameter is a fixed focus position, an optional 'zoom' argument
    can be specified, indicating a fixed focus position to set the device to.
    """
    # TODO Make a template operation, common to 'focus' and 'zoom' operations?

    # Mechanism to convert the actions accepted by the 'zoom' command into Visca codes
    actions2codes = {FOCUS_ACTION_STOP: lambda speed: VISCA_FOCUS_STOP,
                     FOCUS_ACTION_NEAR: lambda
                         speed: VISCA_FOCUS_NEAR if speed is None else VISCA_FOCUS_NEAR_SPEED | speed,
                     FOCUS_ACTION_FAR: lambda
                         speed: VISCA_FOCUS_FAR if speed is None else VISCA_FOCUS_FAR_SPEED | speed}

    if action == FOCUS_ACTION_NEAR or action == FOCUS_ACTION_FAR:
        # Be flexible about the values that can be passed to 'speed',
        # but only when speed is really meaningful
        try:
            # Convert speed into an integer if possible
            speed = int(speed)

            # Make sure it is between the boundaries
            if speed < 0:
                speed = 0
            elif speed > 7:
                speed = 7
        except ValueError as e:
            # We did what we could...
            e.message = "The value of speed should be an integer between 0 and 7"
            raise

    try:
        __cmd_cam(device, VISCA_FOCUS, actions2codes[action](speed), blocking=blocking)
    except KeyError:
        # 'action' is not be a known action (i.e. is not a key in the 'actions2codes' dictionary)
        raise ValueError("'{}' is not a valid focus action".format(action))


def set_focus(device, focus, zoom=None, blocking=False):
    """
    Sets the focus to a fixed position.
    The 'focus' argument is a 4-byte integer. The possible values and their meaning can be consulted in the
    device's manual.
    An optional 'zoom' argument will set the device zoom to the indicated position in one operation.
    'zoom' should be a 4-byte integer, its acceptable values and their meaning to be found at the device's manual.
    """
    if zoom is None:
        __cmd_cam(device, VISCA_FOCUS_VALUE, Packet.int_to_bytes(focus, 4), blocking=blocking)
    else:
        # Sets focus and zoom at the same time
        __cmd_cam(device, VISCA_ZOOM_VALUE, Packet.int_to_bytes(zoom, 4), Packet.int_to_bytes(focus, 4),
                  blocking=blocking)


def set_near_limit(device, limit, blocking=False):
    """
    Set the camera's focus "near limit", i.e. the shortest linear_distance an object can get the focus from the camera.
    The second parameter is a four-byte integer that indicates the near-limit linear_distance.
    Please check the camera manual to see the valid values and their linear_distance equivalence.
    """
    __cmd_cam(device, VISCA_FOCUS_NEAR_LIMIT, Packet.int_to_bytes(limit, 4), blocking=blocking)


def set_focus_mode(device, mode=None, blocking=False):
    """
    Set the focus mode.
    The mode can take any of the following values:
       * None, or not specified, switch the focus mode from auto to manual or vice-versa.
       * 'auto', turn the auto focus on.
       * 'manual', turn the manual focus on.
       * 'trigger', set the "one push trigger" mode.
       * 'infinity', force the camera to focus the infinity
    """
    # TODO Think of a better way to combine this command and the cmd_cam_focus ?
    # TODO Check that a call to this command without parameters will properly change the focus, when it is in the "infinity" mode.

    modes2codes = {None: (VISCA_FOCUS_AUTO, VISCA_FOCUS_AUTO_SWITCH),
                   FOCUS_AUTO_MODE_AUTO: (VISCA_FOCUS_AUTO, VISCA_FOCUS_AUTO_ON),
                   FOCUS_AUTO_MODE_MANUAL: (VISCA_FOCUS_AUTO, VISCA_FOCUS_AUTO_OFF),
                   FOCUS_TRIGGER_MODE_TRIGGER: (VISCA_FOCUS_TRIGGER, VISCA_FOCUS_TRIGGER_TRIGGER),
                   FOCUS_TRIGGER_MODE_INFINITY: (VISCA_FOCUS_TRIGGER, VISCA_FOCUS_TRIGGER_INFINITY)}

    try:
        __cmd_cam(device, *modes2codes[mode], blocking=blocking)
    except KeyError:
        raise ValueError("'{}' is not a valid focus mode".format(mode))


def set_autofocus_low_sensitivity(device, sensitive=True, blocking=False):
    """
    Set the autofocus low sensitivity mode.
    The action performed depends on the second parameter:
       * If it is True (or not specified), set the autofocus sensitivity to 'low'
       * If it is False, set the autofocus sensitivity to 'normal'
    """
    if sensitive:
        __cmd_cam(device, VISCA_FOCUS_AUTO_SENSE, VISCA_FOCUS_AUTO_SENSE_LOW, blocking=blocking)
    else:
        __cmd_cam(device, VISCA_FOCUS_AUTO_SENSE, VISCA_FOCUS_AUTO_SENSE_HIGH, blocking=blocking)


def set_autofocus_mode(device, mode, blocking=False):
    """
    Set the autofocus movement mode
    The action performed depends on the value of the second paramenter:
       * 'normal' - set the autofocus movement mode to "normal".
       * 'interval' - the autofocus movement is only active during a certain interval,
                      as specified by the corresponding command.
       * 'zoom' - the autofocus movement only works during a certain active period
                  (specified by the corresponding command) right after a zoom operation is done.
    """
    modes2codes = {FOCUS_AUTO_MOV_MODE_ZOOM: VISCA_FOCUS_AUTO_MOV_ZOOM,
                   FOCUS_AUTO_MOV_MODE_NORMAL: VISCA_FOCUS_AUTO_MOV_NORMAL,
                   FOCUS_AUTO_MOV_MODE_INTERVAL: VISCA_FOCUS_AUTO_MOV_INTERVAL}

    try:
        __cmd_cam(device, VISCA_FOCUS_AUTO_MOV, modes2codes[mode], blocking=blocking)
    except KeyError:
        raise ValueError("'{}' si not a valid autofocus movement mode".format(mode))


def set_autofocus_active_interval(device, active, interval, blocking=False):
    """
    Set the autofocus 'active' and 'interval' parameters.
    These parameters are relevant for some autofocus modes.
    Both parameters have to be integers between 0x00 and 0xFF
    """
    __cmd_cam(device, VISCA_FOCUS_AUTO_ACTIVE_INT, Packet.int_to_bytes(active, 2), Packet.int_to_bytes(interval, 2),
              blocking=blocking)


def set_ir_correction(device, activate, blocking=False):
    """
    Set the camera correction for IR light.
    The action performed depends on the value of the second parameter:
       * If True, activate the IR light correction
       * If False, deactivate the IR correction
    """
    if activate:
        __cmd_cam(device, VISCA_IR_CORRECTION, VISCA_IR_CORRECTION_ON, blocking=blocking)
    else:
        __cmd_cam(device, VISCA_IR_CORRECTION, VISCA_IR_CORRECTION_OFF, blocking=blocking)


def set_wb_mode(device, mode, blocking=False):
    """
    Set the white balance mode.
    Possible modes are: 'auto', 'indoor', 'outdoor', 'onepush', 'manual'.
    Please check your device's manual for details.
    """
    modes2codes = {WB_AUTO_MODE: VISCA_WB_AUTO,
                   WB_INDOOR_MODE: VISCA_WB_INDOOR,
                   WB_OUTDOOR_MODE: VISCA_WB_OUTDOOR,
                   WB_ONEPUSH_MODE: VISCA_WB_ONEPUSH,
                   WB_MANUAL_MODE: VISCA_WB_MANUAL}

    try:
        __cmd_cam(device, VISCA_WB, modes2codes[mode], blocking=blocking)
    except KeyError:
        raise ValueError("Invalid WB mode")


def trigger_wb(device, blocking=False):
    """
    Trigger the "one push" white balance, when it is set.
    This command makes the camera perform a white balance, when the "one push" WB mode is set
    """
    __cmd_cam(device, VISCA_WB_TRIGGER, VISCA_WB_TRIGGER_ONEPUSH, blocking=blocking)


def set_red_gain(device, action, blocking=False):
    raise NotImplementedError()


def set_blue_gain(device, action, blocking=False):
    raise NotImplementedError()


def set_ae_mode(device, mode, blocking=False):
    """
    Set the auto exposure mode.
    Possible modes are: 'auto', 'manual', 'shutter', 'iris', 'bright'.
    Please check your device's manual for details.
    """
    modes2codes = {AUTO_EXPOSURE_FULL_AUTO_MODE: VISCA_AUTO_EXPOSURE_FULL_AUTO,
                   AUTO_EXPOSURE_MANUAL_MODE: VISCA_AUTO_EXPOSURE_MANUAL,
                   AUTO_EXPOSURE_SHUTTER_PRIORITY_MODE: VISCA_AUTO_EXPOSURE_SHUTTER_PRIORITY,
                   AUTO_EXPOSURE_IRIS_PRIORITY_MODE: VISCA_AUTO_EXPOSURE_IRIS_PRIORITY,
                   AUTO_EXPOSURE_BRIGHT_MODE: VISCA_AUTO_EXPOSURE_BRIGHT}

    try:
        __cmd_cam(device, VISCA_AUTO_EXPOSURE, modes2codes[mode], blocking=blocking)
    except KeyError:
        raise ValueError("Invalid AE mode")


def set_brightness(device, action, blocking=False):
    """
    Set the camera brightness.
    Must set Auto Exposure mode to bright first!
    The action performed depends on the value of the second parameter:
       * If up, increase brightness
       * If down, decrease brightness
    """
    try:
        if action == BRIGHT_ACTION_UP:
            __cmd_cam(device, VISCA_BRIGHT, VISCA_BRIGHT_UP, blocking=blocking)
        elif action == BRIGHT_ACTION_DOWN:
            __cmd_cam(device, VISCA_BRIGHT, VISCA_BRIGHT_DOWN, blocking=blocking)
        else:
            __cmd_cam(device, VISCA_BRIGHT_VALUE, serial.to_bytes("\x00\x00"), Packet.int_to_bytes(action, 2),
                      blocking=blocking)

    except ValueError:
        raise ValueError("The string '{0}', '{1}' or a one-byte-long integer must be passed " \
                         "when adjusting the Brightness".format(BRIGHT_ACTION_UP, BRIGHT_ACTION_DOWN))


def set_exp_comp(device, action, blocking=False):
    """
    Set the camera exposure compensation.
    The action performed depends on the value of the second parameter:
       * If up, increase the Exposure Compensation
       * If down, decrease the Exposure Compensation
    """
    __cmd_cam(device, VISCA_EXPOSURE_COMP, VISCA_EXPOSURE_COMP_ON, blocking=blocking)

    try:
        if action == EXPOSURE_COMP_ACTION_UP:
            __cmd_cam(device, VISCA_EXPOSURE_COMP_AMOUNT, VISCA_EXPOSURE_COMP_UP, blocking=blocking)
        elif action == EXPOSURE_COMP_ACTION_DOWN:
            __cmd_cam(device, VISCA_EXPOSURE_COMP_AMOUNT, VISCA_EXPOSURE_COMP_DOWN, blocking=blocking)
        elif action == EXPOSURE_COMP_ACTION_RESET:
            __cmd_cam(device, VISCA_EXPOSURE_COMP_AMOUNT, VISCA_EXPOSURE_COMP_RESET, blocking=blocking)
        elif action == EXPOSURE_COMP_ACTION_OFF:
            __cmd_cam(device, VISCA_EXPOSURE_COMP, VISCA_EXPOSURE_COMP_OFF, blocking=blocking)

    except ValueError as e:
        e.message = "Select an option {0} {1} {2} or {3}" \
                    "when adjusting the Exposure Compensation".format(EXPOSURE_COMP_ACTION_UP,
                                                                      EXPOSURE_COMP_ACTION_DOWN,
                                                                      EXPOSURE_COMP_ACTION_RESET,
                                                                      EXPOSURE_COMP_ACTION_OFF)
        raise


# Memories
def reset_memory(device, position, blocking=False):
    """
    Reset a memory position.
    Memories store specific camera configurations and positions.
    The second paramenter indicates the memory position to be reset.
    """
    position = int(position)

    if position not in range(MAX_MEMORY_POSITIONS):
        raise ValueError("Invalid memory position: {}".format(position))

    __cmd_cam(device, VISCA_MEMORY, VISCA_MEMORY_RESET, position, blocking=blocking)


def set_memory(device, position, blocking=False):
    """
    Set a memory position with the current configuration and position.
    The second paramenter indicates the memory position to be set.
    """
    position = int(position)

    if position not in range(MAX_MEMORY_POSITIONS):
        raise ValueError("Invalid memory position: {}".format(position))

    __cmd_cam(device, VISCA_MEMORY, VISCA_MEMORY_SET, position, blocking=blocking)


def recall_memory(device, position, blocking=False):
    """
    Recall a memory position.
    Memories store specific camera configurations and positions.
    The second paramenter indicates the memory position to be recalled.
    """
    position = int(position)

    if position not in range(MAX_MEMORY_POSITIONS):
        raise ValueError("Invalid memory position: {}".format(position))

    __cmd_cam(device, VISCA_MEMORY, VISCA_MEMORY_RECALL, position, blocking=blocking)


# Pan and Tilt Drive:
def pan_tilt(device, pan=None, tilt=None, pan_position=None, tilt_position=None, relative=False, blocking=False):
    """
    Specify a pan/tilt movement with the camera.
       * 'pan' indicates the speed of the horizontal movement. It can be positive or negative, so that:
          - A positive speed indicates a movement to the right.
          - A negative speed indicates a movement to the left.
          - Not specified, or zero, indicates that no pan movement must be done.
          - The valid range is device-dependent. Please check the reference manual for details.
       * 'tilt' indicates the speed of the vertical movement. It can be positive or negative, so that:
          - A positive speed indicates a movement upwards.
          - A negative speed indicates a movement downwards.
          - Not specified, or zero, indicates that no tilt movement must be done.
          - The valid range is device-dependent. Please check the reference manual for details.
       * 'pan_position' specifies a fixed horizontal position for the camera.
          - It is an optional parameter.
          - It must be used together with the 'tilt_position' parameter. Cannot be used alone.
          - The camera will move to the indicated horizontal angle at the speed indicated by the 'pan' parameter.
            The speed sign is ignored in this case.
          - The valid range is device-dependent. Please check the reference manual for details.
       * 'tilt_position' specifies a fixed vertical position for the camera.
          - It is an optional parameter.
          - It must be used together with the 'pan_position' parameter. Cannot be used alone.
          - The camera will move to the indicated vertical angle at the speed indicated by the 'tilt' parameter.
            The speed sign is ignored in this case.
          - The valid range is device-dependent. Please check the reference manual for details.
       * 'relative' indicates if the "_position" arguments must be considered absolute or relative to the current position.
         Defaults to "false" (i.e., the positions are considered absolute by default)
    """
    # TODO Use dictionaries
    # TODO Make this command aware of the 'invert' setting?
    # TODO Convert input positions into more intuitive degree intervals

    # Establish default values for parameters pan and tilt
    if pan is None:
        pan = 0
    else:
        # Check pan is within the [-0x7F, 0x7F] range
        # TODO This value has been determined experimentally. May be different in other cameras (it's unlikely, but it's undocumented so we cannot know)
        pan = max(min(pan, 0x7F), -0x7F)

    if tilt is None:
        tilt = 0
    else:
        # Check tilt is within the [-0x7F, 0x7F] range
        # TODO This value has been determined experimentally. May be different in other cameras (it's unlikely, but it's undocumented so we cannot know)
        tilt = max(min(tilt, 0x7F), -0x7F)

    # Convert the parameters to appropriate byte sequences
    if pan:
        if pan > 0:
            horiz = VISCA_PT_DRIVE_HORIZ_RIGHT
        else:
            horiz = VISCA_PT_DRIVE_HORIZ_LEFT
    else:
        horiz = VISCA_PT_DRIVE_HORIZ_STOP

    if tilt:
        if tilt > 0:
            vert = VISCA_PT_DRIVE_VERT_UP
        else:
            vert = VISCA_PT_DRIVE_VERT_DOWN
    else:
        vert = VISCA_PT_DRIVE_VERT_STOP

    if pan_position is not None and tilt_position is not None:
        if relative:
            __cmd_pt(device, VISCA_PT_RELATIVE_POSITION, abs(int(pan)), abs(int(tilt)), \
                     Packet.int_to_bytes(pan_position, 4), Packet.int_to_bytes(tilt_position, 4), blocking=blocking)
        else:
            __cmd_pt(device, VISCA_PT_ABSOLUTE_POSITION, abs(int(pan)), abs(int(tilt)), \
                     Packet.int_to_bytes(pan_position, 4), Packet.int_to_bytes(tilt_position, 4), blocking=blocking)
    elif pan_position is None and tilt_position is None:
        __cmd_pt(device, VISCA_PT_DRIVE, abs(int(pan)), abs(int(tilt)), horiz, vert, blocking=blocking)
    else:
        raise ValueError("Both arguments 'tilt_position' and 'pan_position' must be present or absent at the same time")


def pan_tilt_home(device, blocking=False):
    """
    Move the camera to its "home" position
    """
    __cmd_pt(device, VISCA_PT_HOME, blocking=blocking)


def pan_tilt_reset(device, blocking=False):
    """
    Force the device to recalibrate the pan-tilt position.
    """
    __cmd_pt(device, VISCA_PT_RESET, blocking=blocking)


def osd_off(device, blocking=False):
    """
    Remove OSD info on display
    """
    __cmd_dis(device, VISCA_COMMAND, VISCA_INFO_DISPLAY, VISCA_INFO_DISPLAY_OFF, blocking=blocking)


def osd_on(device, blocking=False):
    """
    Activate OSD info on display
    """
    __cmd_dis(device, VISCA_COMMAND, VISCA_INFO_DISPLAY, VISCA_INFO_DISPLAY_ON, blocking=blocking)


def osd_set(device, on, blocking=False):
    """
    Activates or deactivates the camera's display info
    The action depends on the value of the parameter 'on':
      - on = True: activate OSD
      - on = False: deactivate OSD
    """
    if on:
        osd_on(device, blocking)
    else:
        osd_off(device, blocking)


if __name__ == '__main__':
    import cv2

    cap = cv2.VideoCapture(2)

    # 繰り返しのためのwhile文
    while True:
        # カメラからの画像取得
        ret, frame = cap.read()

        # カメラの画像の出力
        cv2.imshow('camera', frame)

        # 繰り返し分から抜けるためのif文
        key = cv2.waitKey(10)
        if key == 27:
            break

    # メモリを解放して終了するためのコマンド
    cap.release()
    cv2.destroyAllWindows()