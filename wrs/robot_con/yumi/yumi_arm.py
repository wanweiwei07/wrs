"""
Interface of robotic control over ethernet. Built for the YuMi
Author: Weiwei Wan, 20191024, osaka
"""

from multiprocessing import Process, Queue
from queue import Empty
import logging
import socket
import sys
from time import sleep
from collections import namedtuple
import numpy as np
import pickle
import struct
from wrs.robot_con.yumi.autolab_core import RigidTransform
from wrs.robot_con.yumi.yumi_constants import YuMiConstants as YMC
from wrs.robot_con.yumi.yumi_state import YuMiState
from wrs.robot_con.yumi.yumi_motion_logger import YuMiMotionLogger
from wrs.robot_con.yumi.yumi_util import message_to_state, message_to_pose
from wrs.robot_con.yumi.yumi_exceptions import YuMiCommException, YuMiControlException

_RAW_RES = namedtuple('_RAW_RES', 'mirror_code res_code message')
_RES = namedtuple('_RES', 'raw_res data')
_REQ_PACKET = namedtuple('_REQ_PACKET', 'req timeout return_res')

METERS_TO_MM = 1000.0
MM_TO_METERS = 1.0 / METERS_TO_MM


class _YuMiEthernet(Process):

    def __init__(self, req_q, res_q, ip, port, bufsize, timeout, debug):
        Process.__init__(self)

        self._ip = ip
        self._port = port
        self._timeout = timeout
        self._bufsize = bufsize
        self._socket = None

        self._req_q = req_q
        self._res_q = res_q

        self._current_state = None

        self._debug = debug

    def run(self):
        logging.getLogger().setLevel(YMC.LOGGING_LEVEL)

        if self._debug:
            logging.info("In DEBUG mode. Messages will NOT be sent over socket.")
        else:
            self._reset_socket()

        try:
            while True:
                req_packet = self._req_q.get()
                if req_packet == "stop":
                    break
                res = self._send_request(req_packet)
                if req_packet.return_res:
                    self._res_q.put(res)
                sleep(YMC.PROCESS_SLEEP_TIME)

        except KeyboardInterrupt:
            self._stop()
            sys.exit(0)

        self._stop()

    def _stop(self):
        logging.info("Shutting down yumi ethernet interface")
        if not self._debug:
            self._socket.close()

    def _reset_socket(self):
        logging.debug('Opening socket on {0}:{1}'.format(self._ip, self._port))
        if self._socket != None:
            self._socket.close()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._ip, self._port))
        logging.debug('Socket successfully opened!')

    def _send_request(self, req_packet):
        logging.debug("Sending: {0}".format(req_packet))
        raw_res = None

        if self._debug:
            raw_res = '-1 1 MOCK RES for {0}'.format(req_packet)
        else:
            self._socket.settimeout(req_packet.timeout)

            while True:
                try:
                    self._socket.send(req_packet.req)
                    break
                except socket.error as e:
                    # TODO: better way to handle this mysterious bad file descriptor error
                    if e.errno == 9:
                        self._reset_socket()
            try:
                raw_res = self._socket.recv(self._bufsize).decode()
            except socket.error as e:
                if e.errno == 114:  # request time out
                    raise YuMiCommException('Request timed out: {0}'.format(req_packet))

        logging.debug("Received: {0}".format(raw_res))

        if raw_res is None or len(raw_res) == 0:
            raise YuMiCommException('Empty response! For req: {0}'.format(req_packet))

        tokens = raw_res.split()
        res = _RAW_RES(int(tokens[0]), int(tokens[1]), ' '.join(tokens[2:]))
        return res


class YuMiArm:
    """ Interface to a single arm of an ABB YuMi robot_s.
    Communicates with the robot_s over Ethernet.
    """

    def __init__(self, name, ip=YMC.IP, port=YMC.PORTS["left"]["server"], bufsize=YMC.BUFSIZE,
                 motion_timeout=YMC.MOTION_TIMEOUT, comm_timeout=YMC.COMM_TIMEOUT, process_timeout=YMC.PROCESS_TIMEOUT,
                 from_frame='tool', to_frame='base',
                 debug=YMC.DEBUG,
                 log_pose_histories=False, log_state_histories=False,
                 motion_planner=None):
        '''Initializes a YuMiArm interface. This interface will communicate with one arm (port) on the YuMi Robot.
        This uses a subprocess to handle non-blocking socket communication with the RAPID server.

        Parameters
        ----------
            name : string
                    Name of the arm {'left', 'right'}
            ip : string formated ip address, optional
                    IP of YuMi Robot.
                    Default uses the one in YuMiConstants
            port : int, optional
                    Port of target arm's server.
                    Default uses the port for the left arm from YuMiConstants.
            bufsize : int, optional
                    Buffer size for ethernet responses
            motion_timeout : float, optional
                    Timeout for motion commands.
                    Default from YuMiConstants.MOTION_TIMEOUT
            comm_timeout : float, optional
                    Timeout for non-motion ethernet communication.
                    Default from YuMiConstants.COMM_TIMEOUT
            process_timeout : float, optional
                    Timeout for ethernet process communication.
                    Default from YuMiConstants.PROCESS_TIMEOUT
            from_frame : string, optional
                    String name of robot_s arm frame.
                    Default to "tool"
            to_frame : string, optional
                    String name of reference for robot_s frame
                    Default to "base"
            debug : bool, optional
                    Boolean to indicate whether or not in debug mode. If in debug mode no ethernet communication is attempted. Mock responses will be returned.
                    Default to YuMiConstants.DEBUG
            log_pose_histories : bool, optional
                    If True, uses yumi_history_logger to log pose histories. Enables usage of flush_pose_histories.
                    Defaults to False
            log_state_histories : bool, optional
                    If True, uses yumi_history_logger to log state histories. Enables usage of flush_state_histories.
                    Defaults to False
            motion_planner : YuMiMotionPlanner, optional
                    If given, will use for motion trajectories in joint space.
                    Defaults to None
        '''
        self._motion_timeout = motion_timeout
        self._comm_timeout = comm_timeout
        self._process_timeout = process_timeout
        self._ip = ip
        self._port = port
        self._bufsize = bufsize
        self._from_frame = from_frame
        self._to_frame = to_frame
        self._debug = debug

        self._name = name

        if log_pose_histories:
            self._pose_logger = YuMiMotionLogger()
        if log_state_histories:
            self._state_logger = YuMiMotionLogger()

        self.start()

        self._last_sets = {
            'zone': None,
            'speed': None,
            'tool': None,
            'gripper_force': None,
            'gripper_max_speed': None,
        }

        self._motion_planner = motion_planner

    def reset_settings(self):
        '''Reset zone, tool, and speed settings to their last known values. This is used when reconnecting to the RAPID server after a server restart.
        '''
        # set robot_s settings
        for key, val in self._last_sets.items():
            if val is not None:
                getattr(self, 'set_{0}'.format(key))(val)

    def reset(self):
        '''Resets the underlying yumi ethernet process and socket, and resets all the settings.
        '''
        # empty motion logs
        if hasattr(self, '_state_logger'):
            self._state_logger.reset_log()
        if hasattr(self, '_pose_logger'):
            self._pose_logger.reset_log()

        # terminate ethernet
        try:
            self._yumi_ethernet.terminate()
        except Exception:
            pass

        # start ethernet comm
        self._create_yumi_ethernet()

        self.reset_settings()

    def set_motion_planner(self, motion_planner):
        '''
        Parameters
        ----------
        motion_planner : YuMiMotionPlanner
                Sets the current motion planner to the given motion planner.
        '''
        self._motion_planner = motion_planner

    def flush_pose_histories(self, filename):
        '''
        Parameters
        ----------
        filename : string
                Saves the pose history logger data to filename. Empties logger.
        '''
        self._pose_logger.flush_to_file(filename)

    def flush_state_histories(self, filename):
        '''
        Parameters
        ----------
        filename : string
                Saves the state history logger data to filename. Empties logger
        '''
        self._state_logger.flush_to_file(filename)

    def _create_yumi_ethernet(self):
        self._req_q = Queue()
        self._res_q = Queue()

        self._yumi_ethernet = _YuMiEthernet(self._req_q, self._res_q, self._ip, self._port,
                                            self._bufsize, self._comm_timeout, self._debug)
        self._yumi_ethernet.start()

    def start(self):
        '''Starts subprocess for ethernet communication.
        '''
        self._create_yumi_ethernet()

    def stop(self):
        '''Stops subprocess for ethernet communication. Allows program to exit gracefully.
        '''
        self._req_q.put("stop")
        try:
            self._yumi_ethernet.terminate()
        except Exception:
            pass

    def __del__(self):
        self.stop()

    def _request(self, req, wait_for_res, timeout=None):
        if timeout is None:
            timeout = self._comm_timeout

        req_packet = _REQ_PACKET(req, timeout, wait_for_res)
        logging.debug('Process req: {0}'.format(req_packet))

        self._req_q.put(req_packet)
        if wait_for_res:
            try:
                res = self._res_q.get(block=True, timeout=self._process_timeout)
            except (IOError, Empty):
                raise YuMiCommException("Request timed out: {0}".format(req_packet))

            logging.debug('res: {0}'.format(res))

            if res.res_code != YMC.RES_CODES['success']:
                raise YuMiControlException(req_packet, res)

            return res

    @staticmethod
    def _construct_req(code_name, body: str = ""):
        if len(body) < 1:
            _body = []
        else:
            _body = [float(_) for _ in body.split(" ") if _ != ""]
        num_params = len(_body)
        req = struct.pack("HH" + "f" * num_params, num_params, YMC.CMD_CODES[code_name], *_body)
        return req

    @staticmethod
    def _iter_to_str(template, iterable):
        result = ''
        for val in iterable:
            result += template.format(val).rstrip('0').rstrip('.') + ' '
        return result

    @staticmethod
    def _get_pose_body(pose):
        if not isinstance(pose, RigidTransform):
            raise ValueError('Can only parse RigidTransform objects')
        pose = pose.copy()
        pose.position = pose.position * METERS_TO_MM
        body = '{0}{1}'.format(YuMiArm._iter_to_str('{:.1f}', pose.position.tolist()),
                               YuMiArm._iter_to_str('{:.5f}', pose.quaternion.tolist()))
        if pose.configuration is not None:
            body += '{0}'.format(YuMiArm._iter_to_str('{:.5f}', pose.configuration.tolist()))
        return body

    @staticmethod
    def from_frame(self):
        return self._from_frame

    @staticmethod
    def to_frame(self):
        return self._to_frame

    def ping(self, wait_for_res=True):
        '''Pings the remote server.

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        out : Namedtuple (raw_res, data) from ping command.

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('ping')
        return self._request(req, wait_for_res)

    def get_state(self, raw_res=False):
        '''Get the current state (joint configuration) of this arm.

        Parameters
        ----------
        raw_res : bool, optional
                If True, will return raw_res namedtuple instead of YuMiState
                Defaults to False

        Returns
        -------
        out :
            YuMiState if raw_res is False

            _RES(raw_res, state) namedtuple if raw_res is True

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return YuMiState()

        req = YuMiArm._construct_req('get_joints')
        res = self._request(req, True)

        if res is not None:
            state = message_to_state(res.message)
            if raw_res:
                return _RES(res, state)
            else:
                return state

    def get_pose(self, raw_res=False):
        '''Get the current pose of this arm to base frame of the arm.

        Parameters
        ----------
        raw_res : bool, optional
            If True, will return raw_res namedtuple instead of YuMiState
            Defaults to False

        Returns
        -------
        out :
            RigidTransform if raw_res is False

            _RES(raw_res, pose) namedtuple if raw_res is True

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return RigidTransform(from_frame=self._from_frame, to_frame=self._to_frame)

        req = YuMiArm._construct_req('get_pose')
        res = self._request(req, True)

        if res is not None:
            pose = message_to_pose(res.message, self._from_frame)
            if raw_res:
                return _RES(res, pose)
            else:
                return pose

    def is_pose_reachable(self, pose):
        '''Check if a given pose is reachable (incurs no kinematic/joint-space limitations and self collisions)

        Parameters
        ----------
        pose : RigidTransform

        Returns
        -------
        bool : True if pose is reachable, False otherwise.

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = YuMiArm._get_pose_body(pose)
        req = YuMiArm._construct_req('is_pose_reachable', body)
        res = self._request(req, True)
        return bool(int(res.message))

    def goto_state(self, state, wait_for_res=True):
        '''Commands the YuMi to goto the given state (joint angles)

        Parameters
        ----------
        state : YuMiState
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') if state logging is not enabled and wait_for_res is False

        {
            'time': <flaot>,
            'state': <YuMistate>,
            'res': <namedtuple('_RAW_RES', 'mirror_code res_code message')>
        } otherwise. The time field indicates the duration it took for the arm to complete the motion.

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        body = YuMiArm._iter_to_str('{:.2f}', state.joints)
        req = YuMiArm._construct_req('goto_joints', body)
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)

        if hasattr(self, '_state_logger') and wait_for_res and res is not None:
            if self._debug:
                time = -1.
            else:
                time = float(res.message)
            actual_state = self.get_state()
            self._state_logger.append_time(time)
            self._state_logger.append_expected(state)
            self._state_logger.append_actual(actual_state)
            return {
                'time': time,
                'state': actual_state,
                'res': res
            }

        return res

    def contactL(self, state, desired_torque=.1, wait_for_res=True):
        """ YUMi moves linearly toward a configuration. If the torque surpasses the desired torque during the movement, the robot stops.

        Parameters
        state: YuMiState: the target configuration
        desired_torque: num: desired stop torque
        wait_for_res: bool

        Returns
        -------------------
        None if wait_for_res is False
        True if robot reaches the target configuration
        False if robot stops during the movement
        Author: 1/14/2022 Chen Hao
        """
        body = YuMiArm._iter_to_str('{:.2f}', state.jnts + [desired_torque])
        req = YuMiArm._construct_req('contactL', body)
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)
        return bool(int(res.message))

    def fk(self, state, raw_res=False):
        '''Get the forward _kinematics of this arm to base frame of the arm.

        Parameters
        ----------
        state : YuMiState
        raw_res : bool, optional
            If True, will return raw_res namedtuple instead of YuMiState
            Defaults to False

        Returns
        -------
        out :
            RigidTransform if raw_res is False

            _RES(raw_res, pose) namedtuple if raw_res is True

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        if self._debug:
            return RigidTransform(from_frame=self._from_frame, to_frame=self._to_frame)

        body = YuMiArm._iter_to_str('{:.2f}', state.joints)
        req = YuMiArm._construct_req('fk', body)
        res = self._request(req, True)

        if res is not None:
            pose = message_to_pose(res.message, self._from_frame)
            if raw_res:
                return _RES(res, pose)
            else:
                return pose

    def set_speed_max(self, wait_for_res=True):
        'Set speed of YUMI to its max speed'
        req = YuMiArm._construct_req('set_speed_max')
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)
        return

    def _goto_state_sync(self, state, wait_for_res=True):
        body = YuMiArm._iter_to_str('{:.2f}', state.jnts)
        req = YuMiArm._construct_req('goto_joints_sync', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def goto_pose(self, pose, linear=True, relative=False, wait_for_res=True):
        '''Commands the YuMi to goto the given pose

        Parameters
        ----------
        pose : RigidTransform
        linear : bool, optional
            If True, will use MoveL in RAPID to ensure linear path. Otherwise use MoveJ in RAPID, which does not ensure linear path.
            Defaults to True
        relative : bool, optional
            If True, will use goto_pose_relative by computing the delta pose from current pose to target pose.
            Defaults to False
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') if pose logging is not enabled and wait_for_res is False

        {
            'time': <flaot>,
            'pose': <RigidTransform>,
            'res': <namedtuple('_RAW_RES', 'mirror_code res_code message')>
        } otherwise. The time field indicates the duration it took for the arm to complete the motion.

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if relative:
            cur_pose = self.get_pose()
            delta_pose = cur_pose.inverse() * pose
            tra = delta_pose.translation
            rot = np.rad2deg(delta_pose.euler_angles)
            res = self.goto_pose_delta(tra, rot, wait_for_res=wait_for_res)
        else:
            body = YuMiArm._get_pose_body(pose)
            if linear:
                cmd = 'goto_pose_linear'
            else:
                cmd = 'goto_pose'
            req = YuMiArm._construct_req(cmd, body)
            res = self._request(req, wait_for_res, timeout=self._motion_timeout)

        if hasattr(self, '_pose_logger') and wait_for_res and res is not None:
            if self._debug:
                time = -1.
            else:
                time = float(res.message)
            actual_pose = self.get_pose()
            self._pose_logger.append_time(time)
            self._pose_logger.append_expected(pose)
            self._pose_logger.append_actual(actual_pose)
            return {
                'time': time,
                'pose': actual_pose,
                'res': res
            }

        return res

    def _goto_pose_sync(self, pose, wait_for_res=True):
        body = YuMiArm._get_pose_body(pose)
        req = YuMiArm._construct_req('goto_pose_sync', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def goto_pose_linear_path(self, pose, wait_for_res=True,
                              traj_len=10, eef_delta=0.01, jump_thresh=0.0):
        """ Go to a pose via the shortest path in joint space """
        if self._motion_planner is None:
            raise ValueError('Motion motion not enabled')

        current_state = self.get_state()
        current_pose = self.get_pose().as_frames('grippers', 'world')
        traj = self._motion_planner.plan_linear_path(current_state, current_pose,
                                                     pose, traj_len=traj_len,
                                                     eef_delta=eef_delta,
                                                     jump_thresh=jump_thresh)
        if traj is None:
            return
        for state in traj:
            self.goto_state(state, wait_for_res=wait_for_res)

    def goto_pose_shortest_path(self, pose, wait_for_res=True, plan_timeout=0.1):
        """ Go to a pose via the shortest path in joint space """
        if self._motion_planner is None:
            raise ValueError('Motion motion not enabled')

        current_state = self.get_state()
        current_pose = self.get_pose().as_frames('grippers', 'world')
        traj = self._motion_planner.plan_shortest_path(current_state, current_pose,
                                                       pose, timeout=plan_timeout)
        if traj is None:
            return
        for state in traj:
            self.goto_state(state, wait_for_res=wait_for_res)

    def goto_pose_delta(self, translation, rotation=None, wait_for_res=True):
        '''Goto a target pose by transforming the current pose using the given translation and rotation

        Parameters
        ----------
        translation : list-like with axis_length 3
            The translation vector (x, y, z) in meters.
        rotation : list-like with axis_length 3, optional
            The euler angles of given rotation in degrees.
            Defaults to 0 degrees - no rotation.
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        translation = [val * METERS_TO_MM for val in translation]
        translation_str = YuMiArm._iter_to_str('{:.1f}', translation)
        rotation_str = ''
        if rotation is not None:
            rotation_str = YuMiArm._iter_to_str('{:.5f}', rotation)

        body = translation_str + rotation_str
        req = YuMiArm._construct_req('goto_pose_delta', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def set_tool(self, pose, wait_for_res=True):
        '''Sets the Tool Center Point (TCP) of the arm using the given pose.

        Parameters
        ----------
        pose : RigidTransform
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = YuMiArm._get_pose_body(pose)
        req = YuMiArm._construct_req('set_tool', body)
        self._last_sets['tool'] = pose
        return self._request(req, wait_for_res)

    def set_speed(self, speed_data, wait_for_res=True):
        '''Sets the target speed of the arm's movements.

        Parameters
        ----------
        speed_data : list-like with axis_length 4
            Specifies the speed data that will be used by RAPID when executing motions.
            Should be generated using YuMiRobot.get_v
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = YuMiArm._iter_to_str('{:.2f}', speed_data)
        req = YuMiArm._construct_req('set_speed', body)
        self._last_sets['speed'] = speed_data
        return self._request(req, wait_for_res)

    def set_zone(self, zone_data, wait_for_res=True):
        '''Goto a target pose by transforming the current pose using the given translation and rotation

        Parameters
        ----------
        speed_data : list-like with axis_length 4
            Specifies the speed data that will be used by RAPID when executing motions.
            Should be generated using YuMiRobot.get_v
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        pm = zone_data['point_motion']
        data = (pm,) + zone_data['values']
        body = YuMiArm._iter_to_str('{:2f}', data)
        req = YuMiArm._construct_req('set_zone', body)
        self._last_sets['zone'] = zone_data
        return self._request(req, wait_for_res)

    def move_circular(self, center_pose, target_pose, wait_for_res=True):
        '''Goto a target pose by following a circular path around the center_pose

        Parameters
        ----------
        center_pose : RigidTransform
            Pose for the center of the circle for circula movement.
        target_pose : RigidTransform
            Target pose
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        body_set_circ_point = YuMiArm._get_pose_body(center_pose)
        body_move_by_circ_point = YuMiArm._get_pose_body(target_pose)

        req_set_circ_point = YuMiArm._construct_req('set_circ_point', body_set_circ_point)
        req_move_by_circ_point = YuMiArm._construct_req('move_by_circ_point', body_move_by_circ_point)

        res_set_circ_point = self._request(req_set_circ_point, True)
        if res_set_circ_point is None:
            logging.error("Set circular point failed. Skipping move circular!")
            return None
        else:
            return self._request(req_move_by_circ_point, wait_for_res, timeout=self._motion_timeout)

    def movetstate_cont(self, statelist, is_add_all=True, wait_for_res=True) -> bool:
        """
        add states to buffer, execute, and clear

        :param statelist:
        :param wait_for_res:
        :return:

        author: weiwei
        date: 20191024
        Revised by hao chen 20230831
        """
        self.buffer_j_clear(wait_for_res)
        if is_add_all:
            self.buffer_j_add_all2(statelist, wait_for_res=wait_for_res)
        else:
            self.buffer_j_add_all(statelist, wait_for_res=wait_for_res)
        exec_result = self.buffer_j_move(wait_for_res)
        self.buffer_j_clear(wait_for_res)
        return exec_result

    def movetstate_sgl(self, state, wait_for_res=True):
        """
        add states to buffer, execute, and clear

        :param state:
        :param wait_for_res:
        :return:

        author: weiwei
        date: 20191024
        """

        self.goto_state(state, wait_for_res)

    def buffer_c_add_single(self, pose, wait_for_res=True):
        '''Add single pose to the linear movement buffer in RAPID

        Parameters
        ----------
        pose : RigidTransform
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = YuMiArm._get_pose_body(pose)
        req = YuMiArm._construct_req('buffer_c_add', body)
        return self._request(req, wait_for_res)

    def buffer_c_add_all(self, pose_list, wait_for_res=True):
        '''Add a list of poses to the linear movement buffer in RAPID

        Parameters
        ----------
        pose_list : list of RigidTransforms
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        ress = [self.buffer_c_add_single(pose, wait_for_res) for pose in pose_list]
        return ress

    def buffer_c_clear(self, wait_for_res=True):
        '''Clears the linear movement buffer in RAPID

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('buffer_c_clear')
        return self._request(req, wait_for_res)

    def buffer_c_size(self, raw_res=False):
        '''Gets the current linear movement buffer size.

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('buffer_c_size')
        res = self._request(req, True)

        if res is not None:
            try:
                size = int(res.message)
                if raw_res:
                    return _RES(res, size)
                else:
                    return size
            except Exception as e:
                logging.error(e)

    def buffer_c_move(self, wait_for_res=True):
        '''Executes the linear movement buffer

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('buffer_c_move')
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def buffer_j_add_single(self, state, wait_for_res=True):
        '''Add single state to the joint movement buffer in RAPID

        Parameters
        ----------
        state : YuMiState
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = YuMiArm._iter_to_str('{:.2f}', state.joints)
        req = YuMiArm._construct_req('buffer_j_add', body)
        return self._request(req, wait_for_res)

    def buffer_j_add_all(self, state_list, wait_for_res=True):
        '''Add a list of states to the joint movement buffer in RAPID

        Parameters
        ----------
        state_list : list of YuMiState
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        ress = [self.buffer_j_add_single(state, wait_for_res) for state in state_list]
        return ress

    def buffer_j_add_single2(self, state_list, wait_for_res):
        body = YuMiArm._iter_to_str('{:.2f}', [v for state in state_list for v in state.jnts])
        req = YuMiArm._construct_req('buffer_j_add_all', f"{len(state_list)} " + body)
        return self._request(req, wait_for_res)

    def buffer_j_add_all2(self, state_list, msg_len=35, wait_for_res=False):
        ress = [self.buffer_j_add_single2(_, wait_for_res) for _ in
                [state_list[x:x + msg_len] for x in range(0, len(state_list), msg_len)]]
        return ress

    def buffer_j_clear(self, wait_for_res=True):
        '''Clears the joint movement buffer in RAPID

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('buffer_j_clear')
        return self._request(req, wait_for_res)

    def buffer_j_size(self, raw_res=False):
        '''Gets the current joint movement buffer size.

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('buffer_j_size')
        res = self._request(req, True)

        if res is not None:
            try:
                size = int(res.message)
                if raw_res:
                    return _RES(res, size)
                else:
                    return size
            except Exception as e:
                logging.error(e)

    def buffer_j_move(self, wait_for_res=True):
        '''Executes the joint movement buffer

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        req = YuMiArm._construct_req('buffer_j_move')
        res = self._request(req, wait_for_res, timeout=self._motion_timeout)
        return bool(int(res.message))

    def write_handcamimg_ftp(self):
        """
        triger a camera image and send it to the ftp server run on this machine

        :return:

        author: weiwei
        date: 20200519
        """

        req = YuMiArm._construct_req('write_handcamimg_ftp')
        return self._request(req, wait_for_res=True)

    def toggle_vacuum(self, toggletag=True):
        """
        toggle vacuum on and off, toggletag is True or False

        :return:

        author: weiwei
        date: 20200519
        """

        if toggletag:
            req = YuMiArm._construct_req('set_vacuum_on')
            return self._request(req, wait_for_res=True)
        else:
            req = YuMiArm._construct_req('set_vacuum_off')
            return self._request(req, wait_for_res=True)

    def get_pressure(self):
        """
        get the pressure of the vacuum cup

        :return:

        author: weiwei
        date: 20200519
        """

        req = YuMiArm._construct_req('get_pressure')
        res = self._request(req, wait_for_res=True)
        pressure = float(res.message)  # kpa

        return pressure

    def open_gripper(self, no_wait=False, wait_for_res=True):
        '''Opens the grippers to the target_width

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        req = YuMiArm._construct_req('open_gripper', '')
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def close_gripper(self, force=YMC.MAX_GRIPPER_FORCE, width=0., no_wait=False,
                      wait_for_res=True):
        '''Closes the grippers as close to 0 as possible with maximum force.

        Parameters
        ----------
        force : float, optional
            Sets the corresponding gripping force in Newtons.
            Defaults to 20, which is the maximum grip force.
        width : float, optional
            Sets the target width of grippers close motion in m. Cannot be greater than max grippers width.
            Defaults to 0.
        no_wait : bool, optional
            If True, the RAPID server will continue without waiting for the grippers to reach its target width
            Defaults to False
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        if force < 0 or force > YMC.MAX_GRIPPER_FORCE:
            raise ValueError(
                "Gripper force can only be between {} and {}. Got {}.".format(0, YMC.MAX_GRIPPER_FORCE, force))
        if width < 0 or width > YMC.MAX_GRIPPER_WIDTH:
            raise ValueError(
                "Gripper width can only be between {} and {}. Got {}.".format(0, YMC.MAX_GRIPPER_WIDTH, width))

        width = METERS_TO_MM * width
        body = YuMiArm._iter_to_str('{0:.1f}', [force, width] + ([0] if no_wait else []))
        req = YuMiArm._construct_req('close_gripper', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def move_gripper(self, width, no_wait=False, wait_for_res=True):
        '''Moves the grippers to the given width in meters.

        Parameters
        ----------
        width : float
            Target width in meters
        no_wait : bool, optional
            If True, the RAPID server will continue without waiting for the grippers to reach its target width
            Defaults to False
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        lst = [width * METERS_TO_MM]
        if no_wait:
            lst.append(0)
        body = YuMiArm._iter_to_str('{0:.1f}', lst)
        req = YuMiArm._construct_req('move_gripper', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def calibrate_gripper(self, max_speed=None, hold_force=None, phys_limit=None, wait_for_res=True):
        '''Calibrates the grippers.

        Parameters
        ----------
        max_speed : float, optional
            Max speed of the grippers in mm/s.
            Defaults to None. If None, will use maximum speed in RAPID.
        hold_force : float, optional
            Hold force used by the grippers in N.
            Defaults to None. If None, will use maximum force the grippers can provide (20N).
        phys_limit : float, optional
            The maximum opening of the grippers.
            Defaults to None. If None, will use maximum opening the grippers can provide (25mm).
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.

        Notes
        -----
        All 3 values must be provided, or they'll all default to None.
        '''
        if None in (max_speed, hold_force, phys_limit):
            body = ''
        else:
            body = self._iter_to_str('{:.1f}', [data['max_speed'], data['hold_force'], data['phys_limit']])
        req = YuMiArm._construct_req('calibrate_gripper', body)
        return self._request(req, wait_for_res, timeout=self._motion_timeout)

    def set_gripper_force(self, force, wait_for_res=True):
        '''Sets the grippers hold force

        Parameters
        ----------
        force : float
            Hold force by the grippers in N.
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = self._iter_to_str('{:.1f}', [force])
        req = YuMiArm._construct_req('set_gripper_force', body)
        self._last_sets['gripper_force'] = force
        return self._request(req, wait_for_res)

    def set_gripper_max_speed(self, max_speed, wait_for_res=True):
        '''Sets the grippers max speed

        Parameters
        ----------
        max_speed : float
            In mm/s.
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        '''
        body = self._iter_to_str('{:1f}', [max_speed])
        req = YuMiArm._construct_req('set_gripper_max_speed', body)
        self._last_sets['gripper_max_speed'] = max_speed
        return self._request(req, wait_for_res)

    def get_gripper_width(self, raw_res=False):
        '''Get width of current grippers in meters.

        Parameters
        ----------
        raw_res : bool, optional

        Returns
        -------
        Width in meters if raw_res is False
        namedtuple('_RES', 'res width') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        req = YuMiArm._construct_req('get_gripper_width')
        res = self._request(req, wait_for_res=True)

        if self._debug:
            return -1.

        width = float(res.message) * MM_TO_METERS
        if raw_res:
            return _RES(res, width)
        else:
            return width

    def reset_home(self, wait_for_res=True):
        '''Resets the arm to home using joints

        Parameters
        ----------
        wait_for_res : bool, optional
            If True, will block main process until response received from RAPID server.
            Defaults to True

        Returns
        -------
        None if wait_for_res is False
        namedtuple('_RAW_RES', 'mirror_code res_code message') otherwise

        Raises
        ------
        YuMiCommException
            If communication times out or socket error.
        YuMiControlException
            If commanded pose triggers any motion errors that are catchable by RAPID sever.
        '''
        req = YuMiArm._construct_req('reset_home')
        return self._request(req, wait_for_res)


class YuMiArm_ROS:
    """ Interface to remotely control a single arm of an ABB YuMi robot_s.
    Communicates over ROS to a yumi arm server (initialize server through roslaunch)

    Parameters
    ----------
    arm_service : string
        ROSYumiArm service to interface with. If the ROSYumiArm services are started through
        yumi_arms.launch they will be called left_arm and right_arm
    namespace : string, optional
        Namespace to prepend to arm_service. If None, current namespace is prepended.
    """

    def __init__(self, arm_service, namespace=None, timeout=YMC.ROS_TIMEOUT):
        if namespace == None:
            self.arm_service = rospy.get_namespace() + arm_service
        else:
            self.arm_service = namespace + arm_service

        self.timeout = timeout

    def __getattr__(self, name):
        """ Override the __getattr__ method so that function calls become server requests

        If the name is a method of the YuMiArm class, this returns a function that calls that
        function on the YuMiArm instance in the server. The wait_for_res argument is not available
        remotely and will always be set to True. This is to prevent odd desynchronized crashes

        Otherwise, the name is considered to be an attribute, and getattr is called on the
        YuMiArm instance in the server. Note that if it isn't an attribute either a RuntimeError
        will be raised.

        The difference here is that functions access the server *on call* and non-functions do
        *on getting the name*

        Also note that this is __getattr__, so things like __init__ and __dict__ WILL NOT trigger
        this function as the YuMiArm_ROS object already has these as attributes.
        """
        if name in YuMiArm.__dict__:
            def handle_remote_call(*args, **kwargs):
                """ Handle the remote call to some YuMiArm function.
                """
                rospy.wait_for_service(self.arm_service, timeout=self.timeout)
                arm = rospy.ServiceProxy(self.arm_service, ROSYumiArm)
                if 'wait_for_res' in kwargs:
                    kwargs['wait_for_res'] = True
                try:
                    response = arm(pickle.dumps(name), pickle.dumps(args), pickle.dumps(kwargs))
                except rospy.ServiceException as e:
                    raise RuntimeError("Service call failed: {0}".format(str(e)))
                return pickle.loads(response.ret)

            return handle_remote_call
        else:
            rospy.wait_for_service(self.arm_service, timeout=self.timeout)
            arm = rospy.ServiceProxy(self.arm_service, ROSYumiArm)
            try:
                response = arm(pickle.dumps('__getattribute__'), pickle.dumps(name), pickle.dumps(None))
            except rospy.ServiceException as e:
                raise RuntimeError("Could not get attribute: {0}".format(str(e)))
            return pickle.loads(response.ret)


class YuMiArmFactory:
    """ Factory class for YuMiArm interfaces. """

    @staticmethod
    def YuMiArm(arm_type, name, ros_namespace=None):
        """Initializes a YuMiArm interface.

        Parameters
        ----------
        arm_type : string
            Type of arm. One of {'local', 'remote'}

            'local'  creates a local YuMiArm object that communicates over ethernet

            'remote' creates a YuMiArm object that communicates over ROS with a server
        name : string
            Name of arm. One of {'left', 'right'}.

            For local YuMiArm, the port kwarg is set to PORTS[{name}]["server"],
            where PORTS is defined in yumi_constants.py

            For remote YuMiArm, arm_service is set to 'yumi_robot/{name}_arm'.
            This means that the namespace kwarg should be set to the namespace yumi_arms.launch was run in
            (or None if yumi_arms.launch was launched in the current namespace)
        ros_namespace : string
            ROS namespace of arm. Used by remote YuMiArm only.
        """
        if arm_type == 'local':
            return YuMiArm(name, port=YMC.PORTS[name]["server"])
        elif arm_type == 'remote':
            if ROS_ENABLED:
                return YuMiArm_ROS('yumi_robot/{0}_arm'.format(name), namespace=ros_namespace)
            else:
                raise RuntimeError("Remote YuMiArm is not enabled because yumipy is not installed as a catkin package")
        else:
            raise ValueError('YuMiArm end_type {0} not supported'.format(arm_type))


if __name__ == '__main__':
    logging.getLogger().setLevel(YMC.LOGGING_LEVEL)
