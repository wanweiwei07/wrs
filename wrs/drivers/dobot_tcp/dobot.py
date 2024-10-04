"""
The scripts to control the Dobot CR/Nova Robot through TCP/IP protocol
https://github.com/Dobot-Arm/TCP-IP-Protocol/
Author: Hao Chen <chen960216@gmail.com>
Update Notes:
    <20230112>: Create fundamental functions
TODO:
    Add functions for the Modbus and Terminal 485
"""

import re
import time
import socket
import logging
from typing import Literal
from threading import Thread, Condition, Lock
from .dobot_api import DobotApiMove, DobotApiDashboard

import numpy as np

__VERSION__ = (0, 0, 1)

# Each packet received through the real-time feedback port has 1440 bytes
# See for details https://github.com/Dobot-Arm/TCP-IP-Protocol/blob/master/README-EN.md#4-communication-protocolreal--time-feedback-port
PACKET_DATA_TYPE = np.dtype([(
    'len',
    np.int64,
), (
    'digital_input_bits',
    np.uint64,
), (
    'digital_output_bits',
    np.uint64,
), (
    'robot_mode',
    np.uint64,
), (
    'time_stamp',
    np.uint64,
), (
    'time_stamp_reserve_bit',
    np.uint64,
), (
    'test_value',
    np.uint64,
), (
    'test_value_keep_bit',
    np.float64,
), (
    'speed_scaling',
    np.float64,
), (
    'linear_momentum_norm',
    np.float64,
), (
    'v_main',
    np.float64,
), (
    'v_robot',
    np.float64,
), (
    'i_robot',
    np.float64,
), (
    'i_robot_keep_bit1',
    np.float64,
), (
    'i_robot_keep_bit2',
    np.float64,
), ('tool_accelerometer_values', np.float64, (3,)),
    ('elbow_position', np.float64, (3,)),
    ('elbow_velocity', np.float64, (3,)),
    ('q_target', np.float64, (6,)),
    ('qd_target', np.float64, (6,)),
    ('qdd_target', np.float64, (6,)),
    ('i_target', np.float64, (6,)),
    ('m_target', np.float64, (6,)),
    ('q_actual', np.float64, (6,)),
    ('qd_actual', np.float64, (6,)),
    ('i_actual', np.float64, (6,)),
    ('actual_TCP_force', np.float64, (6,)),
    ('tool_vector_actual', np.float64, (6,)),
    ('TCP_speed_actual', np.float64, (6,)),
    ('TCP_force', np.float64, (6,)),
    ('tool_vector_target', np.float64, (6,)),
    ('TCP_speed_target', np.float64, (6,)),
    ('motor_temperatures', np.float64, (6,)),
    ('joint_modes', np.float64, (6,)),
    ('v_actual', np.float64, (6,)),
    # ('dummy', np.float64, (9, 6))])
    ('hand_type', np.byte, (4,)),
    ('user', np.byte,),
    ('tool', np.byte,),
    ('run_queued_cmd', np.byte,),
    ('pause_cmd_flag', np.byte,),
    ('velocity_ratio', np.byte,),
    ('acceleration_ratio', np.byte,),
    ('jerk_ratio', np.byte,),
    ('xyz_velocity_ratio', np.byte,),
    ('r_velocity_ratio', np.byte,),
    ('xyz_acceleration_ratio', np.byte,),
    ('r_acceleration_ratio', np.byte,),
    ('xyz_jerk_ratio', np.byte,),
    ('r_jerk_ratio', np.byte,),
    ('brake_status', np.byte,),
    ('enable_status', np.byte,),
    ('drag_status', np.byte,),
    ('running_status', np.byte,),
    ('error_status', np.byte,),
    ('jog_status', np.byte,),
    ('robot_type', np.byte,),
    ('drag_button_signal', np.byte,),
    ('enable_button_signal', np.byte,),
    ('record_button_signal', np.byte,),
    ('reappear_button_signal', np.byte,),
    ('jaw_button_signal', np.byte,),
    ('six_force_online', np.byte,),
    ('reserve2', np.byte, (82,)),
    ('m_actual', np.float64, (6,)),
    ('load', np.float64,),
    ('center_x', np.float64,),
    ('center_y', np.float64,),
    ('center_z', np.float64,),
    ('user[6]', np.float64, (6,)),
    ('tool[6]', np.float64, (6,)),
    ('trace_index', np.float64,),
    ('six_force_value', np.float64, (6,)),
    ('target_quaternion', np.float64, (4,)),
    ('actual_quaternion', np.float64, (4,)),
    ('reserve3', np.byte, (24,))])
ROBOT_MODE = {'ROBOT_MODE_INIT': 1,
              'ROBOT_MODE_BRAKE_OPEN': 2,
              'ROBOT_MODE_DISABLED': 4,
              'ROBOT_MODE_ENABLE': 5,
              'ROBOT_MODE_BACKDRIVE': 6,
              'ROBOT_MODE_RUNNING': 7,
              'ROBOT_MODE_RECORDING': 8,
              'ROBOT_MODE_ERROR': 9,
              'ROBOT_MODE_PAUSE': 10,
              'ROBOT_MODE_JOG': 11, }
ROBOT_NAME = {3: "CR3",
              31: "CR3L",
              5: "CR5",
              7: "CR7",
              10: "CR10",
              12: "CR12",
              16: "CR16",
              1: "MG400",
              2: "M1Pro",
              101: "Nova 2",
              103: "Nova 5",
              113: "CR3V2",
              115: "CR5V2",
              120: "CR10V2", }


def parse_dobot_return(s):
    error_code, return_val, cmd = re.search('(.+),\{(.*)\},(.+);$', s).groups()
    return int(error_code), return_val


class TimeoutException(Exception):
    def __init__(self, *args):
        super(TimeoutException, self).__init__(*args)


class DobotMonitor(Thread):
    def __init__(self, host: str, monitor_port: Literal[30004, 30005, 30006] = 30004):
        """
        :param host:
        :param monitor_port:  Port 30004 feeds back robot information every 8ms.
                              Port 30005 provides robot information every 200ms.
                              Port 30006 is a configurable port to feed back robot information. By default, port 30006 provides feedback every 50ms.
        """
        super(DobotMonitor, self).__init__()
        self.logger = logging.getLogger("dobot_monitor")
        # connect to the dobot
        try:
            self._socket = socket.create_connection((host, monitor_port), timeout=1)
        except socket.error:
            raise Exception(
                f"Unable to set socket connection use port <{host}:{monitor_port}> !", socket.error)
        self._dict = {}
        self._dict_lock = Lock()
        self._data_event = Condition()
        self._data_queue = bytes()
        self._trystop = False  # to stop thread
        self.running = False  # True when robot_s is on and listening
        self.lastpacket_timestamp = 0
        self.start()
        self.wait()  # make sure we got some data before someone calls us

    def run(self):
        """
        check program execution status in the secondary client data packet we get from the robot_s
        This interface uses only data from the secondary client interface (see UR doc)
        Only the last connected client is the primary client,
        so this is not guaranted and we cannot rely on information to the primary client.
        """
        packet_len = 1440
        while not self._trystop:
            # read data
            tmp_data = self._socket.recv(packet_len)
            # parse data
            recv_data = np.frombuffer(tmp_data, dtype=PACKET_DATA_TYPE)
            # evaluate data
            if hex((recv_data['test_value'][0])) != '0x123456789abcdef':
                continue
            self._data_queue += tmp_data
            with self._dict_lock:
                self._dict = dict(zip(recv_data.dtype.names, recv_data[0]))
                # self._dict = recv_data
            self.lastpacket_timestamp = time.time()
            if self._dict['robot_mode'] == ROBOT_MODE['ROBOT_MODE_RUNNING']:
                self.running = True
            else:
                self.running = False
            with self._data_event:
                self._data_event.notifyAll()

    def wait(self, timeout=1):
        """
        wait for next data packet from robot_s
        """
        tstamp = self.lastpacket_timestamp
        with self._data_event:
            self._data_event.wait(timeout)
            if tstamp == self.lastpacket_timestamp:
                raise TimeoutException("Did not receive a valid data packet from robot_s in {}".format(timeout))

    def get_all_data(self, wait=False):
        """
        return last data obtained from robot_s in dictionnary format
        """
        if wait:
            self.wait()
            time.sleep(.1)
        with self._dict_lock:
            return self._dict.copy()

    def close(self):
        self._trystop = True
        self.join()
        # with self._dataEvent: #wake up any thread that may be waiting for data before we close. Should we do that?
        # self._dataEvent.notifyAll()
        # if self._socket:
        #     with self._prog_queue_lock:
        #         self._s_secondary.close()


class Dobot(object):
    def __init__(self, ip: str, dashboard_port: int = 29999, recv_data_port: Literal[30004, 30005, 30006] = 30004,
                 move_port: int = 30003):
        self.logger = logging.getLogger("dobot")
        self.ip = ip
        self.dobot_mon = DobotMonitor(self.ip, recv_data_port)
        # DobotApi official example
        self.dobot_db = DobotApiDashboard(ip, dashboard_port)
        self.dobot_mov = DobotApiMove(ip, move_port)

    def __repr__(self):
        return f"Robot Object (IP={self.ip}, state={self.dobot_mon.get_all_data()})"

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def robot_mode(self) -> int:
        data = self.dobot_mon.get_all_data()
        return data['robot_mode']

    @property
    def is_enable(self) -> bool:
        mode = self.robot_mode
        return (mode == ROBOT_MODE['ROBOT_MODE_ENABLE']) \
            or (mode == ROBOT_MODE['ROBOT_MODE_RUNNING']) \
            or (mode == ROBOT_MODE['ROBOT_MODE_RUNNING']) \
            or (mode == ROBOT_MODE['ROBOT_MODE_RECORDING'])

    def _wait_for_move(self, target, threshold=None, timeout=5, joints=False):
        """
        wait for a move to complete. Unfortunately there is no good way to know when a move has finished
        so for every received data from robot_s we compute a dist equivalent and when it is lower than
        'threshold' we return.
        if threshold is not reached within timeout, an exception is raised
        """
        self.logger.debug("Waiting for move completion using threshold %s and target %s", threshold, target)
        start_dist = self._get_dist(target, joints)
        if threshold is None:
            threshold = start_dist * 0.8
            if threshold < 0.1:  # roboten precision is limited
                threshold = 0.1
            self.logger.debug("No threshold set, setting it to %s", threshold)
        count = 0
        while True:
            if not self.is_running():
                dist = self._get_dist(target, joints)
                print(dist)
                if dist < threshold:
                    return
                raise Exception("Robot stopped")
            dist = self._get_dist(target, joints)
            self.logger.debug("linear_distance to target is: %s, target dist is %s", dist, threshold)
            if not self.dobot_mon.running:
                if dist < threshold:
                    self.logger.debug("we are threshold(%s) close to target, move has ended", threshold)
                    return
                count += 1
                if count > timeout * 10:
                    raise Exception(
                        "Goal not reached but no program has been running for {} seconds. dist is {}, threshold is {}, ".format(
                            timeout, dist, threshold))
            else:
                count = 0
            time.sleep(.1)

    def _get_dist(self, target, joints=False):
        if joints:
            return self._get_joints_dist(target)
        else:
            return self._get_lin_dist(target)

    def _get_lin_dist(self, target):
        # FIXME: we have an issue here, it seems sometimes the axis angle received from robot_s
        pose = self.get_tcp_cartesian(wait=True)
        dist = 0
        dist += np.sum((target[:3] - pose[:3]) ** 2)
        dist += np.sum(((target[3:6] - pose[3:6]) / 5) ** 2)  # arbitraty axis_length like
        return dist ** 0.5

    def _get_joints_dist(self, target):
        joints = self.get_jnt(wait=True)
        return np.linalg.norm(target - joints)

    def power_on(self):
        """
        Description: Power on the robot. After the robot is powered on, wait about 10 seconds before enabling it.
        """
        self.dobot_db.PowerOn()
        print("Please wait 10 seconds ...")
        time.sleep(10)

    def enable_robot(self):
        """
        Enable the robot
        """
        self.dobot_db.EnableRobot()

    def disable_robot(self):
        """
        Disable the robot
        """
        self.dobot_db.DisableRobot()

    def reset_robot(self):
        """
        Stop the robot
        """
        self.dobot_db.ResetRobot()

    def clear_error(self):
        """
        Clear the error of the robot. After clearing the alarm, the user can judge whether the robot is still in the alarm state according to RobotMode.
        For the alarm that cannot be cleared, restart the control cabinet. (Refer to GetErrorID)
        """
        self.dobot_db.ClearError()

    def is_running(self):
        """
        return True if robot is running
        """
        return self.dobot_mon.running

    def get_robot_type(self, wait=True):
        """
       return name of the robot
       if wait==True, waits for next packet before returning
       """
        data = self.dobot_mon.get_all_data(wait=wait)
        return ROBOT_NAME[data['robot_type']]

    def get_jnt(self, wait=True):
        """
        get joints position
        if wait==True, waits for next packet before returning
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['q_actual']

    def get_jnt_current(self, wait=True):
        """
        get joint currents (j1,j2,j3,j4,j5,j6)
        if wait==True, waits for next packet before returning
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['i_actual']

    def get_tcp_cartesian(self, wait=True):
        """
        get TCP position (x,y,z,rx,ry,rz)
        if wait==True, waits for next packet before returning
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['tool_vector_actual']

    def get_tcp_force(self, wait=True):
        """
        return measured force in TCP (calculated by joint current）
        if wait==True, waits for next packet before returning
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['TCP_force']

    def get_actual_tcp_force(self, wait=True):
        """
        if you do not have a force sensor, you cannot use this function
        return TCP sensor value (calculated by six-axis force)
        if wait==True, waits for next packet before returning
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['actual_TCP_force']

    def get_analog_inputs(self):
        """
        Get the voltage of analog input ports of controller
        """
        return {1: self.get_analog_in(1), 2: self.get_analog_in(2)}

    def get_analog_in(self, index: Literal[1, 2]):
        """
        Get the voltage of analog input port of controller
        """
        error_id, value, cmd = self.dobot_db.AI(index).split(",")
        error_id = int(error_id)
        value = float(value[1: -1])
        return value

    def get_analog_tool_inputs(self):
        """
        Get the voltage of analog input ports of terminal
        """
        return {1: self.get_analog_tool_in(1), 2: self.get_analog_tool_in(2)}

    def get_analog_tool_in(self, index: Literal[1, 2]):
        """
        Get the voltage of analog input port of terminal
        """
        error_id, value, cmd = self.dobot_db.ToolAI(index).split(",")
        error_id = int(error_id)
        value = float(value[1: -1])
        return value

    def get_digital_in_bits(self, wait=True):
        """
        get digital intput (8 bytes)
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['digital_input_bits']

    def get_digital_out_bits(self, wait=True):
        """
        get digital output (8 bytes)
        """
        data = self.dobot_mon.get_all_data(wait=wait)
        return data['digital_output_bits']

    def set_speed(self, ratio: int):
        """
        Set the global speed ratio
        ratio: Speed ratio, range: 0~100, exclusive of 0 and 100
        """
        ratio = min(max(1, ratio), 99)
        self.dobot_db.SpeedFactor(ratio)

    def set_acc_j(self, ratio: int):
        """
         Set the joint acceleration rate. This command is valid only when the motion mode is MovJ, MovJIO, MovJR, JointMovJ
        ratio: Speed ratio, range: 1~100
        """
        ratio = min(max(1, ratio), 100)
        self.dobot_db.AccJ(ratio)

    def set_acc_l(self, ratio: int):
        """
        Set the Cartesian acceleration rate. This command is valid only when the motion mode is MovL, MovLIO, MovLR, Jump, Arc, Circle
        ratio: Speed ratio, range: 1~100
        """
        ratio = min(max(1, ratio), 100)
        self.dobot_db.AccL(ratio)

    def set_payload(self, weight: float, inertia: float):
        """
        set payload in Kg
        inertia in kgm²
        if cog is not specified, then tool center point is used
        """
        return self.dobot_db.PayLoad(weight, inertia)

    def set_analog_out(self, index: Literal[1, 2], val: float, immediate: bool = False):
        """
        set analog output, value is a float
        """
        if immediate:
            self.dobot_db.AOExecute(index, val)
        else:
            self.dobot_db.AO(index, val)

    def set_digital_out(self, index: int, val: Literal[0, 1], immediate: bool = False):
        """
        set digital output. value is a {0,1}
        """
        if immediate:
            self.dobot_db.DOExecute(index, val)
        else:
            self.dobot_db.DO(index, val)

    def set_tool_digital_out(self, index: int, val: Literal[0, 1], immediate: bool = False):
        """
        Set terminal digital output port state.
        """
        if immediate:
            self.dobot_db.ToolDOExecute(index, val)
        else:
            self.dobot_db.ToolDO(index, val)

    def fk(self, jnts: np.ndarray, user: int = 0, tool: int = 0, ):
        """
        Forward _kinematics
        joints: angles of the robot
        user: indicate user frame (Set in the DobotStudio)
        tool: indicate the tool frame (Set in the DobotStudio)
        """
        assert len(jnts) == 6, "The dof of the input joint values must be 6"
        v = self.dobot_db.PositiveSolution(jnts[0], jnts[1], jnts[2], jnts[3], jnts[4], jnts[5],
                                           user, tool)
        error_code, return_val = parse_dobot_return(v)
        if error_code == 0:
            return np.array([float(_) for _ in return_val.split(',')])
        else:
            return None

    def ik(self,
           pos: np.ndarray,
           rot: np.ndarray,
           user: int = 0,
           tool: int = 0,
           seed_jnts: np.ndarray = None):
        """
        Inverse _kinematics
        pos: [x,y,z] position of the TCP
        rotmat: [rx,ry,rz] euler angles of the TCP
        user: indicate user frame (Set in the DobotStudio)
        tool: indicate the tool frame (Set in the DobotStudio)
        """
        if seed_jnts is not None:
            assert len(seed_jnts) == 6, "The dof of the seed joint values must be 6"
            if isinstance(seed_jnts, np.ndarray):
                seed_jnts = seed_jnts.tolist()
        v = self.dobot_db.InverseSolution(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2],
                                          user, tool, seed_jnts)
        error_code, return_val = parse_dobot_return(v)
        if error_code == 0:
            return np.array([float(_) for _ in return_val.split(',')])
        else:
            return None

    def movej(self, jnts: np.ndarray, wait=True):
        """
        move in joint space
        joints: angles of the robot
        """
        assert len(jnts) == 6, "The dof of the input joint values must be 6"
        self.dobot_mov.JointMovJ(jnts[0], jnts[1], jnts[2], jnts[3], jnts[4], jnts[5])
        if wait:
            time.sleep(.5)
            self._wait_for_move(jnts[:6], joints=True)

    def movel(self, pos: np.ndarray, rot: np.ndarray, wait=True):
        """
        linear movement, the target point is Cartesian point
        pos: [x,y,z] position of the TCP
        rotmat: [rx,ry,rz] euler angles of the TCP
        # TODO: add parameters for the speed and acceleration
        """
        self.dobot_mov.MovL(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
        if wait:
            time.sleep(.5)
            self._wait_for_move(np.hstack((pos, rot)), )

    def movep(self, pos: np.ndarray, rot: np.ndarray, wait=True):
        """
        Point to point movement, the target point is Cartesian point
        pos: [x,y,z] position of the TCP
        rotmat: [rx,ry,rz] euler angles of the TCP
        # TODO: add parameters for the speed and acceleration
        """
        # The dobot API name sounds wired.
        self.dobot_mov.MovJ(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])
        if wait:
            time.sleep(.5)
            self._wait_for_move(np.hstack((pos, rot)))

    def servop(self, pos: np.ndarray, rot: np.ndarray, ):
        """
        Dynamic following command based on Cartesian space. You are advised to set the frequency of customer secondary development to 33Hz (30ms), that is, set the cycle interval to at least 30ms.
        pos: [x,y,z] position of the TCP
        rotmat: [rx,ry,rz] euler angles of the TCP
        """
        self.dobot_mov.ServoP(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2])

    def servoj(self, jnts: np.ndarray, ):
        """
        Dynamic following command based on joint space. You are advised to set the frequency of customer secondary development to 33Hz (30ms), that is, set the cycle interval to at least 30ms.
        joints: angles of the robot
        """
        assert len(jnts) == 6, "The dof of the input joint values must be 6"
        self.dobot_mov.ServoJ(jnts[0], jnts[1], jnts[2], jnts[3], jnts[4], jnts[5])

    def close(self):
        """
        close connection to robot_s and stop internal thread
        """
        self.logger.info("Closing sockets to robot_s")
        self.dobot_mon.close()


if __name__ == "__main__":
    rbt = Dobot("192.168.5.1")
    # rbt.power_on()
    # rbt.enable_robot()
    print(rbt.is_enable)
    print(rbt.dobot_mon.get_all_data(wait=True))
