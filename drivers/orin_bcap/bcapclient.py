'''
Software License Agreement (MIT License)

@copyright Copyright (c) 2017 DENSO WAVE INCORPORATED

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, delegator to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

# -*- coding:utf-8 -*-
import select
import socket
import struct
from ctypes import *
from datetime import datetime
from .orinexception import *
from threading import Lock
from .variant import VarType

class BCAPClient:
  _BCAP_SOH = 0x1
  _BCAP_EOT = 0x4

  _TIME_DIFFERENCE = 25569.0
  _SEC_ONEDAY = 24 * 60 * 60

  def datetime2vntdate(date):
    return date.timestamp() \
             / BCAPClient._SEC_ONEDAY + BCAPClient._TIME_DIFFERENCE

  def vntdate2datetime(date):
    return datetime.fromtimestamp( \
             (date - BCAPClient._TIME_DIFFERENCE) * BCAPClient._SEC_ONEDAY)

  _DICT_TYPE2VT  = {
    int        :(VarType.VT_I4  , "i"   , False),
    float      :(VarType.VT_R8  , "d"   , False),
    datetime   :(VarType.VT_DATE, "d"   , False),
    str        :(VarType.VT_BSTR, "I%ds", False),
    bool       :(VarType.VT_BOOL, "h"   , False),
    c_bool     :(VarType.VT_BOOL, "h"   , True),
    c_ubyte    :(VarType.VT_UI1 , "B"   , True),
    c_short    :(VarType.VT_I2  , "h"   , True),
    c_ushort   :(VarType.VT_UI2 , "H"   , True),
    c_int      :(VarType.VT_I4  , "i"   , True),
    c_uint     :(VarType.VT_UI4 , "I"   , True),
    c_long     :(VarType.VT_I4  , "l"   , True),
    c_ulong    :(VarType.VT_UI4 , "L"   , True),
    c_longlong :(VarType.VT_I8  , "q"   , True),
    c_ulonglong:(VarType.VT_UI8 , "Q"   , True),
    c_float    :(VarType.VT_R4  , "f"   , True),
    c_double   :(VarType.VT_R8  , "d"   , True),
    c_wchar_p  :(VarType.VT_BSTR, "I%ds", True),
  }

  _DICT_VT2TYPE = {
    VarType.VT_I2   :("h"  , 2),
    VarType.VT_I4   :("i"  , 4),
    VarType.VT_R4   :("f"  , 4),
    VarType.VT_R8   :("d"  , 8),
    VarType.VT_CY   :("q"  , 8),
    VarType.VT_DATE :("d"  , 8),
    VarType.VT_BSTR :("%ds",-1),
    VarType.VT_ERROR:("i"  , 4),
    VarType.VT_BOOL :("h"  , 2),
    VarType.VT_UI1  :("B"  , 1),
    VarType.VT_UI2  :("H"  , 2),
    VarType.VT_UI4  :("I"  , 4),
    VarType.VT_I8   :("q"  , 8),
    VarType.VT_UI8  :("Q"  , 8),
  }

  def __init__(self, host, port, timeout):
    self._serial  = 1
    self._version = 0
    self._timeout = timeout
    self._sock    = None
    self._lock    = Lock()

    try:
      self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._sock.setblocking(False)
      self._sock.settimeout(timeout)
      self._sock.connect((host, port))
    except OSError as e:
      if not (self._sock is None):
        self._sock.close()
        self._sock = None
      raise e

    self._sock.setblocking(True)

  def __del__(self):
    if not (self._sock is None):
      try:
        self._sock.shutdown(socket.SHUT_RDWR)
      finally:
        self._sock.close()
        self._sock = None

  def settimeout(self, timeout):
    self._timeout = timeout

  def gettimeout(self):
    return self._timeout

  def service_start(self, option = ""):
    self._send_and_recv(1, [option])

  def service_stop(self):
    self._send_and_recv(2, [])

  def controller_connect(self, name, provider, machine, option):
    return self._send_and_recv(3, [name, provider, machine, option])[0]

  def controller_disconnect(self, handle):
    self._send_and_recv(4, [handle])

  def controller_getextension(self, handle, name, option = ""):
    return self._send_and_recv(5, [handle, name, option])[0]

  def controller_getfile(self, handle, name, option = ""):
    return self._send_and_recv(6, [handle, name, option])[0]

  def controller_getrobot(self, handle, name, option = ""):
    return self._send_and_recv(7, [handle, name, option])[0]

  def controller_gettask(self, handle, name, option = ""):
    return self._send_and_recv(8, [handle, name, option])[0]

  def controller_getvariable(self, handle, name, option = ""):
    return self._send_and_recv(9, [handle, name, option])[0]

  def controller_getcommand(self, handle, name, option = ""):
    return self._send_and_recv(10, [handle, name, option])[0]

  def controller_getextensionnames(self, handle, option = ""):
    return self._send_and_recv(11, [handle, option])[0]

  def controller_getfilenames(self, handle, option = ""):
    return self._send_and_recv(12, [handle, option])[0]

  def controller_getrobotnames(self, handle, option = ""):
    return self._send_and_recv(13, [handle, option])[0]

  def controller_gettasknames(self, handle, option = ""):
    return self._send_and_recv(14, [handle, option])[0]

  def controller_getvariablenames(self, handle, option = ""):
    return self._send_and_recv(15, [handle, option])[0]

  def controller_getcommandnames(self, handle, option = ""):
    return self._send_and_recv(16, [handle, option])[0]

  def controller_execute(self, handle, command, param = None):
    return self._send_and_recv(17, [handle, command, param])[0]

  def controller_getmessage(self, handle):
    return self._send_and_recv(18, [handle])[0]

  def controller_getattribute(self, handle):
    return self._send_and_recv(19, [handle])[0]

  def controller_gethelp(self, handle):
    return self._send_and_recv(20, [handle])[0]

  def controller_getname(self, handle):
    return self._send_and_recv(21, [handle])[0]

  def controller_gettag(self, handle):
    return self._send_and_recv(22, [handle])[0]

  def controller_puttag(self, handle, newval):
    self._send_and_recv(23, [handle, newval])

  def controller_getid(self, handle):
    return self._send_and_recv(24, [handle])[0]

  def controller_putid(self, handle, newval):
    self._send_and_recv(25, [handle, newval])

  def extension_getvariable(self, handle, name, option = ""):
    return self._send_and_recv(26, [handle, name, option])[0]

  def extension_getvariablenames(self, handle, option = ""):
    return self._send_and_recv(27, [handle, option])[0]

  def extension_execute(self, handle, command, param = None):
    return self._send_and_recv(28, [handle, command, param])[0]

  def extension_getattribute(self, handle):
    return self._send_and_recv(29, [handle])[0]

  def extension_gethelp(self, handle):
    return self._send_and_recv(30, [handle])[0]

  def extension_getname(self, handle):
    return self._send_and_recv(31, [handle])[0]

  def extension_gettag(self, handle):
    return self._send_and_recv(32, [handle])[0]

  def extension_puttag(self, handle, newval):
    self._send_and_recv(33, [handle, newval])

  def extension_getid(self, handle):
    return self._send_and_recv(34, [handle])[0]

  def extension_putid(self, handle, newval):
    self._send_and_recv(35, [handle, newval])

  def extension_release(self, handle):
    self._send_and_recv(36, [handle])

  def file_getfile(self, handle, name, option = ""):
    return self._send_and_recv(37, [handle, name, option])[0]

  def file_getvariable(self, handle, name, option = ""):
    return self._send_and_recv(38, [handle, name, option])[0]

  def file_getfilenames(self, handle, option = ""):
    return self._send_and_recv(39, [handle, option])[0]

  def file_getvariablenames(self, handle, option = ""):
    return self._send_and_recv(40, [handle, option])[0]

  def file_execute(self, handle, command, param = None):
    return self._send_and_recv(41, [handle, command, param])[0]

  def file_copy(self, handle, name, option = ""):
    self._send_and_recv(42, [handle, name, option])

  def file_delete(self, handle, option = ""):
    self._send_and_recv(43, [handle, option])

  def file_move(self, handle, name, option = ""):
    self._send_and_recv(44, [handle, name, option])

  def file_run(self, handle, option = ""):
    return self._send_and_recv(45, [handle, option])[0]

  def file_getdatecreated(self, handle):
    return self._send_and_recv(46, [handle])[0]

  def file_getdatelastaccessed(self, handle):
    return self._send_and_recv(47, [handle])[0]

  def file_getdatelastmodified(self, handle):
    return self._send_and_recv(48, [handle])[0]

  def file_getpath(self, handle):
    return self._send_and_recv(49, [handle])[0]

  def file_getsize(self, handle):
    return self._send_and_recv(50, [handle])[0]

  def file_gettype(self, handle):
    return self._send_and_recv(51, [handle])[0]

  def file_getvalue(self, handle):
    return self._send_and_recv(52, [handle])[0]

  def file_putvalue(self, handle, newval):
    self._send_and_recv(53, [handle, newval])

  def file_getattribute(self, handle):
    return self._send_and_recv(54, [handle])[0]

  def file_gethelp(self, handle):
    return self._send_and_recv(55, [handle])[0]

  def file_getname(self, handle):
    return self._send_and_recv(56, [handle])[0]

  def file_gettag(self, handle):
    return self._send_and_recv(57, [handle])[0]

  def file_puttag(self, handle, newval):
    self._send_and_recv(58, [handle, newval])

  def file_getid(self, handle):
    return self._send_and_recv(59, [handle])[0]

  def file_putid(self, handle, newval):
    self._send_and_recv(60, [handle, newval])

  def file_release(self, handle):
    self._send_and_recv(61, [handle])

  def robot_getvariable(self, handle, name, option = ""):
    return self._send_and_recv(62, [handle, name, option])[0]

  def robot_getvariablenames(self, handle, option = ""):
    return self._send_and_recv(63, [handle, option])[0]

  def robot_execute(self, handle, command, param = None):
    return self._send_and_recv(64, [handle, command, param])[0]

  def robot_accelerate(self, handle, axis, accel, decel):
    self._send_and_recv(65, [handle, axis, c_float(accel), c_float(decel)])

  def robot_change(self, handle, name):
    self._send_and_recv(66, [handle, name])

  def robot_chuck(self, handle, option = ""):
    self._send_and_recv(67, [handle, option])

  def robot_drive(self, handle, axis, mov, option = ""):
    self._send_and_recv(68, [handle, axis, c_float(mov), option])

  def robot_gohome(self, handle):
    self._send_and_recv(69, [handle])

  def robot_halt(self, handle, option = ""):
    self._send_and_recv(70, [handle, option])

  def robot_hold(self, handle, option = ""):
    self._send_and_recv(71, [handle, option])

  def robot_move(self, handle, comp, pose, option = ""):
    self._send_and_recv(72, [handle, comp, pose, option])

  def robot_rotate(self, handle, rotsuf, deg, pivot, option = ""):
    self._send_and_recv(73, [handle, rotsuf, c_float(deg), pivot, option])

  def robot_speed(self, handle, axis, speed):
    self._send_and_recv(74, [handle, axis, c_float(speed)])

  def robot_unchuck(self, handle, option = ""):
    self._send_and_recv(75, [handle, option])

  def robot_unhold(self, handle, option = ""):
    self._send_and_recv(76, [handle, option])

  def robot_getattribute(self, handle):
    return self._send_and_recv(77, [handle])[0]

  def robot_gethelp(self, handle):
    return self._send_and_recv(78, [handle])[0]

  def robot_getname(self, handle):
    return self._send_and_recv(79, [handle])[0]

  def robot_gettag(self, handle):
    return self._send_and_recv(80, [handle])[0]

  def robot_puttag(self, handle, newval):
    self._send_and_recv(81, [handle, newval])

  def robot_getid(self, handle):
    return self._send_and_recv(82, [handle])[0]

  def robot_putid(self, handle, newval):
    self._send_and_recv(83, [handle, newval])

  def robot_release(self, handle):
    self._send_and_recv(84, [handle])

  def task_getvariable(self, handle, name, option = ""):
    return self._send_and_recv(85, [handle, name, option])[0]

  def task_getvariablenames(self, handle, option = ""):
    return self._send_and_recv(86, [handle, option])[0]

  def task_execute(self, handle, command, param = None):
    return self._send_and_recv(87, [handle, command, param])[0]

  def task_start(self, handle, mode, option = ""):
    self._send_and_recv(88, [handle, mode, option])

  def task_stop(self, handle, mode, option = ""):
    self._send_and_recv(89, [handle, mode, option])

  def task_delete(self, handle, option = ""):
    self._send_and_recv(90, [handle, option])

  def task_getfilename(self, handle):
    return self._send_and_recv(91, [handle])[0]

  def task_getattribute(self, handle):
    return self._send_and_recv(92, [handle])[0]

  def task_gethelp(self, handle):
    return self._send_and_recv(93, [handle])[0]

  def task_getname(self, handle):
    return self._send_and_recv(94, [handle])[0]

  def task_gettag(self, handle):
    return self._send_and_recv(95, [handle])[0]

  def task_puttag(self, handle, newval):
    self._send_and_recv(96, [handle, newval])

  def task_getid(self, handle):
    return self._send_and_recv(97, [handle])[0]

  def task_putid(self, handle, newval):
    self._send_and_recv(98, [handle, newval])

  def task_release(self, handle):
    self._send_and_recv(99, [handle])

  def variable_getdatetime(self, handle):
    return self._send_and_recv(100, [handle])[0]

  def variable_getvalue(self, handle):
    return self._send_and_recv(101, [handle])[0]

  def variable_putvalue(self, handle, newval):
    self._send_and_recv(102, [handle, newval])

  def variable_getattribute(self, handle):
    return self._send_and_recv(103, [handle])[0]

  def variable_gethelp(self, handle):
    return self._send_and_recv(104, [handle])[0]

  def variable_getname(self, handle):
    return self._send_and_recv(105, [handle])[0]

  def variable_gettag(self, handle):
    return self._send_and_recv(106, [handle])[0]

  def variable_puttag(self, handle, newval):
    self._send_and_recv(107, [handle, newval])

  def variable_getid(self, handle):
    return self._send_and_recv(108, [handle])[0]

  def variable_putid(self, handle, newval):
    self._send_and_recv(109, [handle, newval])

  def variable_getmicrosecond(self, handle):
    return self._send_and_recv(110, [handle])[0]

  def variable_release(self, handle):
    self._send_and_recv(111, [handle])

  def command_execute(self, handle, mode):
    self._send_and_recv(112, [handle, mode])

  def command_cancel(self, handle):
    self._send_and_recv(113, [handle])

  def command_gettimeout(self, handle):
    return self._send_and_recv(114, [handle])[0]

  def command_puttimeout(self, handle, newval):
    self._send_and_recv(115, [handle, newval])

  def command_getstate(self, handle):
    return self._send_and_recv(116, [handle])[0]

  def command_getparameters(self, handle):
    return self._send_and_recv(117, [handle])[0]

  def command_putparameters(self, handle, newval):
    self._send_and_recv(118, [handle, newval])

  def command_getresult(self, handle):
    return self._send_and_recv(119, [handle])[0]

  def command_getattribute(self, handle):
    return self._send_and_recv(120, [handle])[0]

  def command_gethelp(self, handle):
    return self._send_and_recv(121, [handle])[0]

  def command_getname(self, handle):
    return self._send_and_recv(122, [handle])[0]

  def command_gettag(self, handle):
    return self._send_and_recv(123, [handle])[0]

  def command_puttag(self, handle, newval):
    self._send_and_recv(124, [handle, newval])

  def command_getid(self, handle):
    return self._send_and_recv(125, [handle])[0]

  def command_putid(self, handle, newval):
    self._send_and_recv(126, [handle, newval])

  def command_release(self, handle):
    self._send_and_recv(127, [handle])

  def message_reply(self, handle, data):
    self._send_and_recv(128, [handle, data])

  def message_clear(self, handle):
    self._send_and_recv(129, [handle])

  def message_getdatetime(self, handle):
    return self._send_and_recv(130, [handle])[0]

  def message_getdescription(self, handle):
    return self._send_and_recv(131, [handle])[0]

  def message_getdestination(self, handle):
    return self._send_and_recv(132, [handle])[0]

  def message_getnumber(self, handle):
    return self._send_and_recv(133, [handle])[0]

  def message_getserialnumber(self, handle):
    return self._send_and_recv(134, [handle])[0]

  def message_getsource(self, handle):
    return self._send_and_recv(135, [handle])[0]

  def message_getvalue(self, handle):
    return self._send_and_recv(136, [handle])[0]

  def message_release(self, handle):
    self._send_and_recv(137, [handle])

  def _send_and_recv(self, funcid, args):
    with self._lock:
      self._bcap_send(self._serial, self._version, funcid, args)
      (serial, version, hresult, retvals) = self._bcap_recv()

      if self._serial >= 0xFFFF:
        self._serial  = 1
      else:
        self._serial += 1

      if HResult.failed(hresult):
        raise ORiNException(hresult)

    if len(retvals) == 0:
      retvals.append(None)
      
    return retvals

  def _bcap_send(self, serial, version, funcid, args):
    buf = self._serialize(serial, version, funcid, args)
    flags = 0
    if hasattr(socket, 'MSG_NOSIGNAL'):
      flags |= socket.MSG_NOSIGNAL
    self._sock.sendall(buf, flags)

  def _serialize(self, serial, version, funcid, args):
    format = "<bIHhiH"
    packet_data = [BCAPClient._BCAP_SOH, 0, serial, version, funcid, len(args)]

    packed_args = self._serialize_args(args, True)

    format += "%ds" % len(packed_args)
    packet_data.append(packed_args)

    format += "b"
    packet_data.append(BCAPClient._BCAP_EOT)

    buf = struct.pack(format, *packet_data)
    buf = buf.replace(b'\0\0\0\0', struct.pack("<I", len(buf)), 1)

    return buf

  def _serialize_args(self, args, first = False):
    format = "<"
    packet_data = []
    offset = 0

    for arg in args:
      if first:
        format += "I"
        packet_data.append(0)

      packed_arg = self._serialize_arg(arg)
      len_arg = len(packed_arg)
      format += "%ds" % len_arg
      packet_data.append(packed_arg)

      if first:
        packet_data[2*offset] = len_arg

      offset += 1

    if len(packet_data) > 0:
      return struct.pack(format, *packet_data)
    else:
      return b''

  def _serialize_arg(self, arg):
    format = "<HI"
    packet_data = []

    if arg is None:
      packet_data = [VarType.VT_EMPTY, 1]

    elif isinstance(arg, (list, tuple)):
      len_arg = len(arg)

      if len_arg <= 0:
        packet_data = [VarType.VT_EMPTY, 1]
      else:
        is_vntary = False
        type_o0 = type(arg[0])
        for o in arg:
          if type_o0 != type(o):
            is_vntary = True
            break

        if is_vntary:
          packed_args = self._serialize_args(arg)
          format += "%ds" % len(packed_args)
          packet_data += [VarType.VT_VARIANT | VarType.VT_ARRAY, len_arg, packed_args]        

        else:
          if type_o0 in BCAPClient._DICT_TYPE2VT:
            (vt, fmt, is_ctype) = BCAPClient._DICT_TYPE2VT[type_o0]

            if vt == VarType.VT_DATE:
              format += fmt * len_arg
              packet_data += [vt | VarType.VT_ARRAY, len_arg]
              for o in arg:
                packet_data.append(BCAPClient.datetime2vntdate(o))

            elif vt == VarType.VT_BSTR:
              packet_data += [vt | VarType.VT_ARRAY, len_arg]
              for o in arg:
                if is_ctype:
                  str_tmp = o.value.encode("utf-16le")
                else:
                  str_tmp = o.encode("utf-16le")
                len_str = len(str_tmp)
                format += fmt % len_str
                packet_data += [len_str, str_tmp]

            elif vt == VarType.VT_BOOL:
              format += fmt * len_arg
              packet_data += [vt | VarType.VT_ARRAY, len_arg]
              for o in arg:
                if o:
                  packet_data.append(-1)
                else:
                  packet_data.append(0)

            else:
              format += fmt * len_arg
              packet_data += [vt | VarType.VT_ARRAY, len_arg]
              if is_ctype:
                for o in arg:
                  packet_data.append(o.value)
              else:
                packet_data += arg

          else:
            raise ORiNException(HResult.E_CAO_VARIANT_TYPE_NOSUPPORT)

    elif isinstance(arg, (bytes, bytearray)):
      len_arg = len(arg)
      format += "%ds" % len_arg
      packet_data += [VarType.VT_ARRAY | VarType.VT_UI1, len_arg, arg]

    else:
      type_arg = type(arg)
      if type_arg in BCAPClient._DICT_TYPE2VT:
        (vt, fmt, is_ctype) = BCAPClient._DICT_TYPE2VT[type_arg]

        if vt == VarType.VT_DATE:
          format += fmt
          date_tmp = BCAPClient.datetime2vntdate(arg)
          packet_data += [vt, 1, date_tmp]

        elif vt == VarType.VT_BSTR:
          if is_ctype:
            str_tmp = arg.value.encode("utf-16le")
          else:
            str_tmp = arg.encode("utf-16le")
          len_str = len(str_tmp)
          format += fmt % len_str
          packet_data += [vt, 1, len_str, str_tmp]

        elif vt == VarType.VT_BOOL:
          format += fmt
          if arg:
            packet_data += [vt, 1, -1]
          else:
            packet_data += [vt, 1,  0]

        else:
          format += fmt
          if is_ctype:
            packet_data += [vt, 1, arg.value]
          else:
            packet_data += [vt, 1, arg]

      else:
        raise ORiNException(HResult.E_CAO_VARIANT_TYPE_NOSUPPORT)

    return struct.pack(format, *packet_data)

  def _bcap_recv(self):
    while True:
      buf_all = b''

      buf_tmp = self._recv_with_select(1)
      buf_all = b''.join([buf_all, buf_tmp])

      buf_tmp  = self._recv_with_select(4)
      len_recv = struct.unpack("<I", buf_tmp)
      buf_all  = b''.join([buf_all, buf_tmp])

      buf_tmp = self._recv_with_select(len_recv[0] - 5)
      buf_all = b''.join([buf_all, buf_tmp])

      (serial, version, hresult, retvals) = self._deserialize(buf_all)

      if (self._serial == serial) and (hresult != HResult.S_EXECUTING):
        break

    return (serial, version, hresult, retvals)

  def _recv_with_select(self, len_recv):
    buf_recv = b''
    while True:
      (reads, writes, errors) = select.select(
        [self._sock], [], [], self._timeout)

      if len(reads) == 0:
        raise ORiNException(HResult.E_TIMEOUT)

      buf_recv = b''.join([buf_recv,
        self._sock.recv(len_recv)])

      if len(buf_recv) >= len_recv:
        break

    return buf_recv

  def _deserialize(self, buf):
    format = "<bIHhiH%dsb" % (len(buf) - 16)
    (soh, len_buf, serial, version, hresult, len_args, buf_args, eot) \
      = struct.unpack(format, buf)

    if (soh != BCAPClient._BCAP_SOH) or (eot != BCAPClient._BCAP_EOT):
      raise ORiNException(HResult.E_INVALIDPACKET)

    (retvals, buf_args) = self._deserialize_args(buf_args, len_args, True)

    return (serial, version, hresult, retvals)

  def _deserialize_args(self, buf, len_args, first = False):
    retvals = []

    for i in range(len_args):
      if first:
        buf = buf[4:]
      (retval, buf) = self._deserialize_arg(buf)
      retvals.append(retval)

    return (retvals, buf)

  def _deserialize_arg(self, buf):
    retval = None

    format = "<HI%ds" % (len(buf) - 6)
    (vt, len_arg, buf) = struct.unpack(format, buf)

    if (vt & VarType.VT_ARRAY) != 0:
      vt = vt ^ VarType.VT_ARRAY
      if vt == VarType.VT_VARIANT:
        (retval, buf) = self._deserialize_args(buf, len_arg)

      elif vt == VarType.VT_UI1:
        format = "<%ds%ds" % (len_arg, len(buf) - len_arg)
        (retval, buf) = struct.unpack(format, buf)

      elif vt in BCAPClient._DICT_VT2TYPE:
        (fmt, len_val) = BCAPClient._DICT_VT2TYPE[vt]

        if vt == VarType.VT_BSTR:
          retval = []
          for i in range(len_arg):
            format = "<I%ds" % (len(buf) - 4)
            (len_str, buf) = struct.unpack(format, buf)
            format = "<%ds%ds" % (len_str, len(buf) - len_str)
            (ret_tmp, buf) = struct.unpack(format, buf)
            retval.append(ret_tmp.decode("utf-16le"))

        else:
          format = "<%s%ds" % (fmt * len_arg, len(buf) - (len_val * len_arg))
          unpacked_arg = struct.unpack(format, buf)
          retval   = list(unpacked_arg[:-1])
          buf      = unpacked_arg[-1]

          if vt == VarType.VT_DATE:
            for i in range(len(retval)):
              retval[i] = BCAPClient.vntdate2datetime(retval[i])

          elif vt == VarType.VT_BOOL:
            for i in range(len(retval)):
              retval[i] = (retval[i] != 0)

      else:
        raise ORiNException(HResult.E_CAO_VARIANT_TYPE_NOSUPPORT)

    else:
      if vt in [ VarType.VT_EMPTY, VarType.VT_NULL ]:
        pass
      elif vt in BCAPClient._DICT_VT2TYPE:
        (fmt, len_val) = BCAPClient._DICT_VT2TYPE[vt]
        
        if vt == VarType.VT_BSTR:
          format = "<I%ds" % (len(buf) - 4)
          (len_str, buf) = struct.unpack(format, buf)
          format = "<%ds%ds" % (len_str, len(buf) - len_str)
          (retval , buf) = struct.unpack(format, buf)
          retval = retval.decode("utf-16le")

        else:
          format = "<%s%ds" % (fmt, (len(buf) - len_val))
          (retval, buf) = struct.unpack(format, buf)

          if vt == VarType.VT_DATE:
            retval = BCAPClient.vntdate2datetime(retval)

          elif vt == VarType.VT_BOOL:
            retval = (retval != 0)

      else:
        raise ORiNException(HResult.E_CAO_VARIANT_TYPE_NOSUPPORT)

    return (retval, buf)
