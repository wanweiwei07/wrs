"""
A high-level API for the dynamixel based grippers for the sdk_wrapper.py. Two motors are used to control the grippers.
Author: Chen Hao (chen960216@gmail.com), 20221015, osaka
Update Notes:
    -`0.0.1`: Implement the basic function

TODO:
    - Currently, sync write is only support for the goal postion. Add more support in future
    - Test Linux support
"""
import os
import json
import time
from typing import Literal
import numpy as np
from wrs.drivers.devices.dynamixel_sdk.sdk_wrapper import DynamixelMotor

# get the path of the current file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DATA_FILE_NAME = os.path.join(THIS_DIR, "nova2_gripper_x_calibration_data.json")
SAVE_CALIBRATION_DATETIME = "NOVA2_GRIPPER_X_SAVE_CALIBRATION_DATETIME"
IS_CALIBRATED_NAME = "NOVA2_GRIPPER_X_IS_CALIBRATED"
CLOSE_GRIPPER_MOTOR_POS_NAME = "NOVA2_GRIPPER_X_CLOSE_GRIPPER_MOTOR_POS"
OPEN_GRIPPER_MOTOR_POS_NAME = "NOVA2_GRIPPER_X_OPEN_GRIPPER_MOTOR_POS"


class Nova2GripperX(object):
    def __init__(self,
                 device: str = "COM7",
                 motor_ids: tuple or list = (1, 2),
                 baudrate: int = 115200,
                 op_mode: int = 5,
                 gripper_limit: tuple or list = (0, 0.01),
                 motor_max_current: int = 170,
                 load_calibration_data: bool = True):
        """
        Initialize the Nova2 grippers
        :param device: The device name of the serial port
        :param motor_ids: dynaxmiel motor ids of the grippers
        :param baudrate: The baudrate of the serial port
        :param op_mode: The operation mode of the dynamixel motor
                    0	Current Control Mode
                    1	Velocity Control Mode
                    3	Position Control Mode
                    4   Extended Position Control Mode (Multi-turn)
                    5   Current-based Position Control Mode
                    16  PWM Control Mode
        :param gripper_limit: The grippers limit, (min, max)
        :param motor_max_current: The max current of the motor (See the manual for details)
        :param load_calibration_data: If True, load the calibration data from the file
        """
        assert motor_ids is not None, "Please specify the motor ids."
        assert isinstance(motor_ids, (tuple, list)), "The motor ids should be a tuple or list."
        assert gripper_limit is not None, "Please specify the grippers limit."
        assert isinstance(gripper_limit, (tuple, list)), "The grippers limit should be a tuple or list."
        assert isinstance(motor_max_current, int), "The motor max current should be an integer."
        assert isinstance(baudrate, int), "The baudrate should be an integer."
        self._motor_x = DynamixelMotor(device=device, baud_rate=baudrate, toggle_group_sync_write=True)
        self._motor_ids = motor_ids
        # Set the operation mode
        for motor_id in motor_ids:
            if self._motor_x.get_dxl_op_mode(motor_id) != op_mode:
                self._motor_x.disable_dxl_torque(motor_id)
                self._motor_x.set_dxl_op_mode(op_mode=op_mode, dxl_id=motor_id)
        # Enable the torque
        [self._motor_x.enable_dxl_torque(motor_id) for motor_id in motor_ids]
        # Set the current limit
        [self._motor_x.set_dxl_goal_current(motor_max_current, motor_id) for motor_id in motor_ids]
        # Set the grippers limit
        self._gripper_limit = gripper_limit
        # read the calibration data
        self.is_calibrated = False
        self._close_gripper_motor_pos = None
        self._open_gripper_motor_pos = None
        if load_calibration_data and os.path.exists(CALIBRATION_DATA_FILE_NAME):
            with open(CALIBRATION_DATA_FILE_NAME, "r") as f:
                env = json.load(f)
                if SAVE_CALIBRATION_DATETIME in env:
                    save_calibration_datetime = env[SAVE_CALIBRATION_DATETIME]
                if time.time() - save_calibration_datetime < 3600 * 24 * 10:  # 10 days to update the calibration data
                    if IS_CALIBRATED_NAME in env:
                        self.is_calibrated = env[IS_CALIBRATED_NAME]
                    if CLOSE_GRIPPER_MOTOR_POS_NAME in env:
                        self._close_gripper_motor_pos = env[CLOSE_GRIPPER_MOTOR_POS_NAME]
                    if OPEN_GRIPPER_MOTOR_POS_NAME in env:
                        self._open_gripper_motor_pos = env[OPEN_GRIPPER_MOTOR_POS_NAME]

    @property
    def is_calibrated(self) -> bool:
        """
        If the grippers is calibrated
        """
        return self._is_calibrated

    @is_calibrated.setter
    def is_calibrated(self, value: bool):
        """
        Set the calibration flag
        """
        assert isinstance(value, bool), "The value should be a boolean."
        self._is_calibrated = value

    # TODO write the equation to exactly calucate the relationship between the grippers width and the motor position
    def map_gripper_width_to_motor_pos(self, width: float or int) -> list:
        """
        Map the grippers width to the motor position
        :param width: The width of the grippers
        :return: The motor position
        """
        assert self.is_calibrated, "Please calibrate the grippers first."
        assert isinstance(width, (float, int)), "The width should be a number."
        assert self._gripper_limit[0] - 0.01 <= width <= self._gripper_limit[
            1] + 0.01, "The width is out of the grippers limit."
        # Calculate the motor position
        motor_pos = [int(self._close_gripper_motor_pos[i] + (
                self._open_gripper_motor_pos[i] - self._close_gripper_motor_pos[i]) * width / (
                                 self._gripper_limit[1] - self._gripper_limit[0]))
                     for i, motor_id in enumerate(self._motor_ids)]
        return motor_pos

    def calibrate(self, close_gripper_direction: Literal[-1, 1] = 1, speed: int = 100):
        """
        Calibrate the grippers. The grippers will be closed and opened to get the motor position.
        :param close_gripper_direction: The motion_vec of the grippers when closing the grippers
        :param speed: The speed of the grippers
        """
        assert close_gripper_direction in [-1, 1], "The close_gripper_direction should be -1 or 1."
        # Set the speed of the motor
        [self._motor_x.set_dxl_position_p_gain(speed, motor_id) for motor_id in self._motor_ids]
        time.sleep(.1)
        # Set the goal current of the grippers
        [self._motor_x.set_dxl_goal_current(200, motor_id) for motor_id in self._motor_ids]
        time.sleep(.1)
        motor_locs = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self._motor_ids]
        if close_gripper_direction == 1:
            # close the grippers
            [self._motor_x.set_dxl_goal_pos(
                min(motor_locs[i] + 4000, self._motor_x._control_table.DXL_MAX_POSITION_VAL)
                , motor_id) for i, motor_id in enumerate(self._motor_ids)]
        else:
            [self._motor_x.set_dxl_goal_pos(
                max(motor_locs[i] - 4000, self._motor_x._control_table.DXL_MIN_POSITION_VAL)
                , motor_id) for i, motor_id in enumerate(self._motor_ids)]
        time.sleep(.1)
        while np.any([self._motor_x.is_moving(motor_id) for motor_id in self._motor_ids]):
            time.sleep(.1)
        # get the current position
        motor_close_pos = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self._motor_ids]
        time.sleep(.5)
        # Get the open position
        if close_gripper_direction == 1:
            # close the grippers
            [self._motor_x.set_dxl_goal_pos(max(motor_locs[i] - 4000, self._motor_x._control_table.DXL_MIN_POSITION_VAL)
                                            , motor_id) for i, motor_id in enumerate(self._motor_ids)]
        else:
            [self._motor_x.set_dxl_goal_pos(min(motor_locs[i] + 4000, self._motor_x._control_table.DXL_MAX_POSITION_VAL)
                                            , motor_id) for i, motor_id in enumerate(self._motor_ids)]
        time.sleep(.1)
        while np.any([self._motor_x.is_moving(motor_id) for motor_id in self._motor_ids]):
            time.sleep(.1)
        # get the current position
        motor_open_pos = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self._motor_ids]
        self._close_gripper_motor_pos = motor_close_pos
        self._open_gripper_motor_pos = motor_open_pos
        self.is_calibrated = True  # set the calibration flag
        self.save_calibration_data()

    def save_calibration_data(self):
        """
        Save the calibration data to the file
        """
        data = dict()
        data[SAVE_CALIBRATION_DATETIME] = time.time()
        data[IS_CALIBRATED_NAME] = self.is_calibrated
        data[CLOSE_GRIPPER_MOTOR_POS_NAME] = self._close_gripper_motor_pos
        data[OPEN_GRIPPER_MOTOR_POS_NAME] = self._open_gripper_motor_pos
        with open(CALIBRATION_DATA_FILE_NAME, "w") as f:
            json.dump(data, f)

    def set_gripper_width(self, width: float or int, speed: int = 200, grasp_force: int = 170,
                          wait: bool = True, disable_motor_id: int = None) -> bool:
        """
        Set the grippers width
        :param width: The width of the grippers
        :param speed: The speed of the grippers
        :param grasp_force: The grasp force of the grippers (Described by current)
        :param wait: If True, wait until the grippers is stopped
        :return: True if the grippers is stopped
        """
        if not self.is_calibrated:
            raise ValueError("Please calibrate the grippers first.")
        assert isinstance(width, (float, int)), "The width should be a number."
        assert isinstance(speed, int), "The speed should be an integer."
        assert isinstance(grasp_force, int), "The grasp force should be an integer."
        assert self._gripper_limit[0] - 0.01 <= width <= self._gripper_limit[
            1] + 0.01, "The width is out of the grippers limit."
        [self._motor_x.set_dxl_position_p_gain(speed, motor_id) for motor_id in self._motor_ids]
        time.sleep(.1)
        # Set the goal current of the grippers
        [self._motor_x.set_dxl_goal_current(grasp_force, motor_id) for motor_id in self._motor_ids]
        time.sleep(.1)
        # ret = [self._motor_x.set_dxl_goal_pos(self.map_gripper_width_to_motor_pos(width)[i], self._motor_ids[i])
        #        for i, motor_id in enumerate(self._motor_ids) if disable_motor_id != motor_id]
        if disable_motor_id is not None:
            tgt_pos = self.map_gripper_width_to_motor_pos(width)
            tgt_pos_tmp = []
            motor_ids_tmp = []
            for i, motor_id in enumerate(self._motor_ids):
                if motor_id == disable_motor_id:
                    continue
                tgt_pos_tmp.append(tgt_pos[i])
                motor_ids_tmp.append(motor_id)
            ret = self._motor_x.set_dxl_goal_pos_sync(tgt_pos_tmp, motor_ids_tmp)
        else:
            ret = self._motor_x.set_dxl_goal_pos_sync(self.map_gripper_width_to_motor_pos(width), self._motor_ids)
        if wait:
            time.sleep(.1)
            while np.any([self._motor_x.is_moving(motor_id) for motor_id in self._motor_ids]):
                time.sleep(.1)
        return np.all(ret)

    def get_gripper_width(self) -> float:
        """
        Get the grippers width
        :return: The width of the grippers
        """
        if not self.is_calibrated:
            raise ValueError("Please calibrate the grippers first.")
        motor_pos = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self._motor_ids]
        width = (motor_pos[0] - self._close_gripper_motor_pos[0]) / (
                self._open_gripper_motor_pos[0] - self._close_gripper_motor_pos[0]) * (
                        self._gripper_limit[1] - self._gripper_limit[0]) + self._gripper_limit[0]
        return width

    def open_gripper(self, speed: int = 200, grasp_force=170, wait: bool = True, disable_motor_id: int = None) -> bool:
        """
        Open the grippers
        """
        return self.set_gripper_width(self._gripper_limit[1] + 0.01, speed, grasp_force, wait, disable_motor_id)

    def close_gripper(self, speed: int = 200, grasp_force=170, wait=True, disable_motor_id: int = None) -> bool:
        """
        Close the grippers
        """
        return self.set_gripper_width(self._gripper_limit[0] + 0.01, speed, grasp_force, wait, disable_motor_id)

    def __del__(self):
        """
        Disable the torque of the motor
        :return: None
        """
        try:
            [self._motor_x.disable_dxl_torque(motor_id) for motor_id in self._motor_ids]
        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        gp = Nova2GripperX(device='COM8', motor_ids=[1, 2], baudrate=115200, op_mode=5, gripper_limit=(0, 0.198),
                           motor_max_current=170, )
        if not gp.is_calibrated:
            gp.calibrate(close_gripper_direction=-1)
        else:
            print("The grippers is calibrated.")
        gp.open_gripper()
        # gpa.open_gripper()
        input("Press to finish...")
        # gpa.close_gripper(grasp_force=170)
        # time.sleep(3)
        # gpa.close_gripper(grasp_force=200)
        # input("Press Enter to continue...")
        # gpa.close_gripper(grasp_force=80, disable_motor_id=2)
        # gpa.close_gripper()
        gp.set_gripper_width(width=0.0, grasp_force=30)
        gp.set_gripper_width(width=0.1, grasp_force=30)
        input("Press to finish...")
    except KeyboardInterrupt:
        del gp
    except Exception as e:
        print(e)
