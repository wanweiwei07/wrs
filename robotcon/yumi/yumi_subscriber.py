'''
Class for streaming pose and state information from the YuMi
Author: Jacky Liang
'''

from multiprocessing import Process, Queue
from queue import Empty
import socket
import sys
from collections import namedtuple
import logging
from time import sleep
from setproctitle import setproctitle
from robotconn.yumirapid.yumi_constants import YuMiConstants as YMC
from robotconn.yumirapid.yumi_state import YuMiState
from robotconn.yumirapid.yumi_exceptions import YuMiCommException
from robotconn.yumirapid.yumi_util import message_to_state, message_to_pose, message_to_torques

_RAW_RES = namedtuple('_RAW_RES', 'time message')

class _YuMiSubscriberEthernet(Process):

    def __init__(self, name, data_q, cmd_q, ip, port, bufsize, timeout):
        Process.__init__(self)
        self._name = name
        self._data_q = data_q
        self._cmd_q = cmd_q

        self._ip = ip
        self._port = port
        self._bufsize = bufsize
        self._timeout = timeout

        self._end_run = False

        self._socket = None

    def run(self):
        setproctitle('python._YuMiSubscriberEthernet.{0}'.format(self._name))
        logging.getLogger().setLevel(YMC.LOGGING_LEVEL)

        self._reset_socket()
        try:
            while True:
                if self._cmd_q.qsize() > 0:
                    cmd = self._cmd_q.get()
                    if cmd == 'stop':
                        break
                    if cmd == 'reset':
                        self._reset_socket()
                else:
                    raw_res = None
                    try:
                        raw_res = self._socket.recv(self._bufsize)
                    except socket.error as e:
                        if e.errno == 114: # request time out
                            raise YuMiCommException('Request timed out')

                    if raw_res is None or len(raw_res) == 0:
                        self._stop()
                        break

                    raw_res = raw_res[:raw_res.rfind("!")]
                    raw_res = raw_res[raw_res.rfind("#")+1:]
                    tokens = raw_res.split()
                    res = _RAW_RES(float(tokens[0]), ' '.join(tokens[1:]))

                    if self._data_q.full():
                        try:
                            self._data_q.get_nowait()
                        except Empty:
                            pass

                    self._data_q.put(res)
                sleep(1e-3)
        except KeyboardInterrupt:
            pass
        self._stop()
        sys.exit(0)

    def _stop(self):
        logging.debug("Shutting down yumi subscriber ethernet interface for {0}".format(self._name))
        self._socket.close()

    def _reset_socket(self):
        logging.debug('Opening socket on {0}:{1} for subscription'.format(self._ip, self._port))
        if self._socket != None:
            self._socket.close()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._ip, self._port))
        logging.debug('Socket successfully opened!')

class _YuMiArmSubscriber:

    def __init__(self, name, includes):
        self._name = name
        self._to_frame = "yumi_{0}".format(name)
        self._comm_timeout = YMC.COMM_TIMEOUT
        self._bufsize = YMC.BUFSIZE
        self._ip = YMC.IP
        self._time_offset = 0
        self._qs = {
            'cmd': {},
            'data': {}
        }
        self.includes = includes
        for name in self.includes:
            self._qs['cmd'][name] = Queue()
            self._qs['data'][name] = Queue(maxsize=1)
        self.msgs_map = {
            'states': message_to_state,
            'torques': message_to_torques,
            'poses': message_to_pose
        }

    def _start(self):
        self._last_datas = {}
        self._subs = {}
        for name in self.includes:
            self._last_datas[name] = None
            self._subs[name] = _YuMiSubscriberEthernet("{}_{}".format(self._name, name),
                                                            self._qs['data'][name],
                                                            self._qs['cmd'][name],
                                                            self._ip, YMC.PORTS[self._name][name],
                                                            self._bufsize,
                                                            self._comm_timeout)
            self._subs[name].start()

    def _stop(self):
        for q in self._qs['cmd'].values():
            q.put('stop')

    def __del__(self):
        self._stop()

    def _reset_time(self):
        self._time_offset += self.get_pose()[0]

    def _get_data(self, name, timestamp=True):
        if name not in self.includes:
            raise ValueError("Subscriber wasn't instantiated to include {}!".format(name))

        data_q = self._qs['data'][name]
        if not data_q.empty() or self._last_datas[name] is None:
            res = data_q.get(block=True)
            self._last_datas[name] = res.time, self.msgs_map[name](res.message)

        time_stamp, data = self._last_datas[name]

        if timestamp:
            return time_stamp - self._time_offset, data
        return data

    def get_pose(self, timestamp=True):
        '''Get the current pose of this arm.

        Parameters
        ----------
        timestamp : bool, optional
            Returns timestamp along with the pose if True
            Defaults to True

        Returns
        -------
        out : float (seconds since start or last call to reset, whichever one is more recent), RigidTransform

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        return self._get_data('poses', timestamp=timestamp)

    def get_state(self, timestamp=True):
        '''Get the current state (joint configuration) of this arm and corresponding timestamp.

        Parameters
        ----------
        timestamp : bool, optional
            Returns timestamp along with the state if True
            Defaults to True

        Returns
        -------
        out : float (seconds since start or last call to reset, whichever one is more recent), YuMiState

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        return self._get_data('states', timestamp=timestamp)

    def get_torque(self, timestamp=True):
        '''Get the current torque readings of each joint of this arm and corresponding timestamp.

        Parameters
        ----------
        timestamp : bool, optional
            Returns timestamp along with the torque if True
            Defaults to True

        Returns
        -------
        out : float (seconds since start or last call to reset, whichever one is more recent), numpy array

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        return self._get_data('torques', timestamp=timestamp)

class YuMiSubscriber:
    """ Interface to stream pose and state information from an ABB YuMI robot_s
    over ethernet.
    """

    def __init__(self, include_left=True, include_right=True,
                        left_includes=('torques', 'states', 'poses'),
                        right_includes=('torques', 'states', 'poses')):
        '''Initializes a YuMiSubscriber

        Parameters
        ----------
            include_left : bool, optional
                    If True, the left arm subscriber is included and instantiated.
                    Defaults to True
            include_right : bool, optional
                    If True, the right arm subscriber is included and instantiated.
                    Defaults to True

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        if not include_left and not include_right:
            raise ValueError("Must include at least one arm to subscribe to!")

        self._arms = {}
        if include_left:
            self.left = _YuMiArmSubscriber("left", left_includes)
            self._arms['left'] = self.left
        if include_right:
            self.right = _YuMiArmSubscriber("right", right_includes)
            self._arms['right'] = self.right
        self._started = False

    def stop(self):
        '''Calls the stop function for each instantiated arm subscriber object.
        '''
        if not self._started:
            raise Exception("Cannot stop a YuMiSubscriber if it hasn't started yet!")

        for sub in self._arms.values():
            sub._stop()
        self._started = False

    def start(self):
        '''Calls the start function for each instantiated arm subscriber object.
        '''
        if self._started:
            raise Exception("Can only start a YuMiSubscriber once!")

        self._started = True
        for sub in self._arms.values():
            sub._start()

    def reset_time(self):
        if not self._started:
            raise Exception("Can only reset time if YuMiSubscriber has started!")

        for sub in self._arms.values():
            sub._reset_time()
