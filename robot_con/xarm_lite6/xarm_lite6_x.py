"""
WRS control interface for XArm Lite 6
Author: Chen Hao (chen960216@gmail.com), 20220912, osaka
Reference: XArm Developer Manual (http://download.ufactory.cc/xarm/en/xArm%20Developer%20Manual.pdf?v=1600992000052)
           XArm Python SDK (https://github.com/xArm-Developer/xArm-Python-SDK)
"""
import time
from typing import Optional

import numpy as np

import basis.robot_math as rm
import drivers.xarm.wrapper.xarm_api as arm
from xarm_lite6_dxl_x import XArmLite6DXLCon

try:
    import motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

__VERSION__ = '0.0.3'


class XArmLite6X(object):
    def __init__(self, ip: str = "192.168.1.232", reset: bool = False, has_gripper=False):
        """
        :param ip: The ip address of the robot
        """
        # examine parameters
        # assert isinstance('ip', str) and mode in ['position', 'servo']
        assert isinstance('ip', str)
        # initialization
        self._arm_x = arm.XArmAPI(port=ip)
        driver_v = self._arm_x.version_number
        # ensure the xarm driver is larger than 1.9.0 https://github.com/xArm-Developer/xArm-Python-SDK#1910
        assert driver_v >= (1, 9, 0)
        # reset error
        if self._arm_x.has_err_warn:
            err_code = self._arm_x.get_err_warn_code()[1][0]
            if err_code == 1 or err_code == 2:
                print("The Emergency Button is pushed in to stop!")
                input("Release the emergency button and press any key to continue. Press Enter to continue...")
        self._arm_x.clean_error()
        self._arm_x.motion_enable()
        if reset:
            self._arm_x.reset(wait=True)
        else:
            self._arm_x.set_state(0)
        time.sleep(.5)

        # for gripper
        if has_gripper:
            self._gripper_x = XArmLite6DXLCon(self._arm_x, baudrate=115200, dxl_id=2)
            self._gripper_x.enable_dxl_torque()
            self._gripper_limit = [0, 0.04]
            if self._gripper_x.get_dxl_op_mode() != 5:
                # enter current based position control mode
                self._gripper_x.set_dxl_op_mode(5)
            # set up gripper's parameters
            self._max_current = 100  # torque related parameter
            self._gripper_x.set_dxl_goal_current(self._max_current)
            self._is_gripper_calibrated = False
            self._zero_dxl_pos = None

        self.ndof = 6

    @staticmethod
    def pos_unit_xarm2wrs(arr: np.ndarray) -> np.ndarray:
        """
        Convert the position in XArm API to the WRS system
        :param arr: Position array obtained from the XArm API
        :return: Converted position array
        """
        return arr / 1000

    @staticmethod
    def pos_unit_wrs2xarm(arr: np.ndarray) -> np.ndarray:
        """
        Convert the position in WRS system to the XArm API
        :param arr: Position array in the WRS system
        :return: Converted position array
        """
        return arr * 1000

    @staticmethod
    def pos_unit_dxl2wrs(pos: int) -> float:
        """
        Conver the position of Dynaxmiel motor to the WRS system
        :param pos: pos of the Dynamixel Motor
        :return: pos of the WRS system
        """
        gear_circum = np.pi * 25 / 1000
        return gear_circum * 0.0879 / 360 * pos

    @staticmethod
    def pos_unit_wrs2dxl(pos: float, dxl_bias: int = 0) -> int:
        """
        Convert the WRS system position to the Dynamixel motor position
        :param pos: pos of the WRS system
        :return: pos of the Dynamixel Motor
        """
        gear_circum = np.pi * 25 / 1000
        return int(pos / (gear_circum * 0.0879 / 360)) + dxl_bias

    @property
    def mode(self) -> int:
        """
        xArm mode, only available in socket way and  enable_report is True
        :return:  0: position control mode
                  1: servo motion mode
                  2: joint teaching mode
                  3: cartesian teaching mode (invalid)
                  4: joint velocity control mode
                  5: cartesian velocity control mode

        """
        return self._arm_x.mode

    @property
    def state(self) -> int:
        """
        Get the state of the robot
        :return: tuple((code, state)), only when code is 0, the returned result is correct.
            code: See the [API Code Documentation](./xarm_api_code.md#api-code) for details.
            state:
                1: in motion
                2: sleeping
                3: suspended
                4: stopping
        """
        return self._arm_x.get_state()

    @property
    def cmd_num(self) -> int:
        """
        Get the cmd count in cache
        :return: tuple((code, cmd_num)), only when code is 0, the returned result is correct.
            code: See the [API Code Documentation](./xarm_api_code.md#api-code) for details.
        """
        code, cmd_num = self._arm_x.cmd_num
        self._ex_ret_code(code)
        return cmd_num

    def _ex_ret_code(self, code):
        """
        Examine the return code of the instruction. If the code is not 0 (success), a Exception will be raised.
        :param code:
        :return:
        """
        if code != 0:
            raise Exception(f"The return code {code} is incorrect. Refer API for details")

    def _position_mode(self):
        """
        Enter the position control mode
        """
        if self.mode != 0:
            self._arm_x.arm.set_mode(0)
            self._arm_x.arm.set_state(state=0)
            time.sleep(.5)

    def _servo_mode(self):
        """
        Enter the servo motion mode
        """
        if self.mode != 1:
            self._arm_x.arm.set_mode(1)
            self._arm_x.arm.set_state(state=0)
            time.sleep(.5)

    def reset(self):
        self._arm_x.reset()

    def homeconf(self):
        self.move_j(jnt_val=np.array([0., 0.173311, 0.555015, 0., 0.381703, 0.]), )

    def ik(self, tgt_pos: np.ndarray, tgt_rot: np.ndarray) -> np.ndarray:
        """

        :param tgt_pos: The position under WRS system
        :param tgt_rot: The 3x3 Rotation matrix or 1x3 RPY matrix
        :return: inverse kinematics solution
        """
        tgt_pos = self.pos_unit_wrs2xarm(tgt_pos)
        if tgt_rot is not None:
            if tgt_rot.shape == (3, 3):
                tgt_rpy = rm.rotmat_to_euler(tgt_rot)
            else:
                tgt_rpy = tgt_rot.flatten()[:3]
        tgt_pose = tgt_pos.tolist() + tgt_rpy.tolist()
        code, ik_s = self._arm_x.get_inverse_kinematics(pose=tgt_pose, input_is_radian=True, return_is_radian=True)
        self._ex_ret_code(code)
        return np.array(ik_s)

    def get_gripper_width(self) -> float:
        if not self._is_gripper_calibrated or self._zero_dxl_pos is None:
            raise Exception("Calibrate gripper first")
        dxl_pos = self._gripper_x.get_dxl_pos()
        print(dxl_pos, self._zero_dxl_pos)
        return self.pos_unit_dxl2wrs(dxl_pos - self._zero_dxl_pos) * 2

    def set_gripper_width(self, width: float, speed: int = 800, wait: bool = True) -> bool:
        if not self._is_gripper_calibrated or self._zero_dxl_pos is None:
            raise Exception("Calibrate gripper first")
        assert self._gripper_limit[0] <= width <= self._gripper_limit[1]
        self._gripper_x.set_dxl_position_p_gain(speed)
        ret = self._gripper_x.set_dxl_goal_pos(self.pos_unit_wrs2dxl(width / 2, dxl_bias=self._zero_dxl_pos))
        if wait:
            while self._gripper_x.is_moving():
                time.sleep(.1)
        return ret

    def open_gripper(self, speed: int = 800, wait: bool = True) -> bool:
        return self.set_gripper_width(self._gripper_limit[1], speed, wait)

    def close_gripper(self, speed: int = 800, wait=True) -> bool:
        return self.set_gripper_width(self._gripper_limit[0], speed, wait)

    def manual_calibrate_gripper(self):
        """
        Manually calibrate the gripper. Only run this functions when the gear and racks on the gripper are separated.
        Process:
            1. The motor will first go to 360deg. Then go back to 180deg.
            2. Then, wait a second and the motor will exit torque mode and the human can make the gear and rack into contact.
        :return:
        """
        self._gripper_x.set_dxl_position_p_gain(100)
        print(self._gripper_x.get_dxl_pos())
        # self._gripper_x.set_dxl_goal_pos(4095)
        self._gripper_x.set_dxl_goal_pos(4095)
        time.sleep(.5)
        while self._gripper_x.is_moving():
            time.sleep(.1)
        self._gripper_x.set_dxl_goal_pos(2000)
        while self._gripper_x.is_moving():
            time.sleep(.1)
        self._gripper_x.disable_dxl_torque()
        input("Manually setup the gripper_1")
        self._gripper_x.enable_dxl_torque()
        self._gripper_x.set_dxl_goal_current(10)
        time.sleep(.1)
        self._gripper_x.set_dxl_goal_pos(0)
        while self._gripper_x.is_moving():
            time.sleep(.1)
        self._zero_dxl_pos = self._gripper_x.get_dxl_pos()
        print("The zero position is ", self._zero_dxl_pos)
        self._gripper_x.set_dxl_goal_current(self._max_current)
        self._is_gripper_calibrated = True

    def calibrate_gripper(self):
        self._gripper_x.set_dxl_position_p_gain(100)
        self._gripper_x.set_dxl_goal_current(10)
        time.sleep(.1)
        self._gripper_x.set_dxl_goal_pos(0)
        while self._gripper_x.is_moving():
            time.sleep(.1)
        self._zero_dxl_pos = self._gripper_x.get_dxl_pos()
        print("The zero position is ", self._zero_dxl_pos)
        self._gripper_x.set_dxl_goal_current(self._max_current)
        self._is_gripper_calibrated = True

    def get_jnt_values(self) -> np.ndarray:
        """
        Get the joint values of the arm
        :return: Joint values (Array)
        """
        code, jnt_val = self._arm_x.get_servo_angle(is_radian=True)
        jnt_val = jnt_val[:self.ndof]
        self._ex_ret_code(code)
        return np.array(jnt_val)

    def get_pose(self) -> (np.ndarray, np.ndarray):
        """
        Get the cartesian position
        :return: tuple(Position(Array), Orientation(Array))
        """
        code, pose = self._arm_x.get_position(is_radian=True)
        self._ex_ret_code(code)
        return self.pos_unit_xarm2wrs(np.array(pose[:3])), rm.rotmat_from_euler(*pose[3:])

    def move_j(self,
               jnt_val: np.ndarray,
               speed: Optional[float] = None,
               is_rel_mov: bool = False,
               wait: bool = True) -> bool:
        """
        Move the robot to a target joint value
        :param jnt_val: Targe joint value (1x6 Array)
        :param speed: Move speed (rad/s)
        :param is_rel_mov: Relative move or not
        :param wait: whether to wait for the arm to complete, default is True
        :return: if the path is moved successfully, it will return 0
        """
        if isinstance(jnt_val, np.ndarray):
            jnt_val = jnt_val.tolist()
        assert isinstance(jnt_val, list) and len(jnt_val) == self.ndof
        self._position_mode()
        suc = self._arm_x.set_servo_angle(angle=jnt_val, speed=speed, is_radian=True,
                                          relative=is_rel_mov, wait=wait)
        if suc == 0:
            return True
        else:
            return False

    def move_p(self,
               pos: Optional[np.ndarray],
               rot: Optional[np.ndarray],
               speed: Optional[float] = None,
               path_rad: Optional[float] = None,
               is_rel_mov: bool = False,
               wait: bool = True) -> bool:
        """
        Move to a pose under the robot base coordinate
        :param pos: Position (Array([x,y,z])) of the pose
        :param rot: Orientation (Array([roll,pitch,yaw]) or Array(3x3)) of the pose
        :param speed: Move speed (mm/s, rad/s)
        :param path_rad: move radius, if radius is larger or equal than 0, will MoveArcLine, else MoveLine
        :param is_rel_mov: Relative move or not
        :return: if the path is moved successfully, it will return 0
        :param wait: whether to wait for the arm to complete, default is True
        """
        assert pos is not None or rot is not None
        assert path_rad is None or path_rad >= 0
        self._position_mode()
        if pos is not None:
            pos = np.array(pos)
            pos = self.pos_unit_wrs2xarm(pos)
        else:
            pos = [None] * 3
        if rot is not None:
            rot = np.array(rot)
            if rot.shape == (3, 3):
                rpy = rm.rotmat_to_euler(rot)
            else:
                rpy = rot.flatten()[:3]
        else:
            rpy = [None] * 3
        suc = self._arm_x.set_position(x=pos[0], y=pos[1], z=pos[2],
                                       roll=rpy[0], pitch=rpy[1], yaw=rpy[2], speed=speed,
                                       is_radian=True,
                                       relative=is_rel_mov,
                                       radius=path_rad,
                                       wait=wait)
        if suc == 0:
            return True
        else:
            return False

    def move_jntspace_path(self, path,
                           max_jntvel: list = None,
                           max_jntacc: list = None,
                           start_frame_id=1,
                           toggle_debug=False):
        """
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :param max_jntvel: 1x6 list to describe the maximum joint speed for the arm
        :param max_jntacc: 1x6 list to describe the maximum joint acceleration for the arm
        :param start_frame_id:
        :return:
        author: weiwei
        """
        if TOPPRA_EXIST:
            # enter servo mode
            self._servo_mode()
            if not path or path is None:
                raise ValueError("The given is incorrect!")
            # Refer to https://www.ufactory.cc/_files/ugd/896670_9ce29284b6474a97b0fc20c221615017.pdf
            # the robotic arm can accept joint position commands sent at a fixed high frequency like 100Hz
            control_frequency = .05
            tpply = pwp.PiecewisePolyTOPPRA()
            interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                                control_frequency=control_frequency,
                                                                max_vels=max_jntvel,
                                                                max_accs=max_jntacc,
                                                                toggle_debug=False)
            interpolated_path = interpolated_path[start_frame_id:]
            for jnt_values in interpolated_path:
                self._arm_x.set_servo_angle_j(jnt_values, is_radian=True)
                time.sleep(.05)
            return
        else:
            raise NotImplementedError

    def __del__(self):
        self._arm_x.disconnect()
        self._gripper_x.disable_dxl_torque()
