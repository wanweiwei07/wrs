"""
Yumi Higher-level Control API.
Author: Hao Chen
Date: 20220123
"""
import time
from typing import Tuple, List

import numpy as np

import robot_con.yumi.yumi_robot as yr
import robot_con.yumi.yumi_state as ys


class Yumi_Controller:
    def __init__(self, debug: bool = False):
        """
        is_add_all: Set True, the function `move_jntspace_path` will send multiple joint angles at once.
                    Otherwise, it will send single joint angle at once
        """
        self.rbtx = yr.YuMiRobot(debug=debug)
        self._is_add_all = True

    @property
    def lft_arm_hnd(self):
        return self.rbtx.left

    @property
    def rgt_arm_hnd(self):
        return self.rbtx.right

    def get_pose(self, component_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get pose of the robot computed by YUMI Server
        :return 1x3 position vector and 3x3 rotation matrix
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        pose = armx.get_pose()
        pos = pose.translation
        rot = pose.rotation
        return pos, rot

    def move_jnts(self, component_name: str, jnt_vals: np.ndarray, speed_n: int = 100):
        """
        move one arm joints of the yumi
        :param component_name
        :param jnt_vals: 1x7 np.array
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        if speed_n == -1:
            armx.set_speed_max()
        else:
            self.rbtx.set_v(speed_n)
        armjnts = np.rad2deg(jnt_vals)
        ajstate = ys.YuMiState(armjnts)
        armx.movetstate_sgl(ajstate)

    def contactL(self, component_name: str, jnt_vals: np.ndarray, desired_torque: float = .5) -> bool:
        """
        Use contactL. Move the robot to a target pose. The robot will stop in advance if the torque reach desired torque
        :return True if the robot reach target pose else False
        """
        if component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['rgt_arm']!")
        armjnts = np.rad2deg(jnt_vals)
        ajstate = ys.YuMiState(armjnts)
        return armx.contactL(ajstate, desired_torque)

    def get_jnt_values(self, component_name: str):
        """
        get the joint angles of both arms
        :return: 1x6 array
        """
        if component_name == "all":
            lftjnts = self._get_arm_jnts("lft")
            rgtjnts = self._get_arm_jnts("rgt")
            return np.array(lftjnts + rgtjnts)
        elif component_name in ["lft_arm", "lft_hnd"]:
            return self._get_arm_jnts("lft")
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            return self._get_arm_jnts("rgt")
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")

    def move_jntspace_path(self, component_name: str, path: List[np.ndarray], speed_n: int = 100) -> bool:
        """
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        statelist = []
        st = time.time()
        for armjnts in path:
            armjnts = np.rad2deg(armjnts)
            ajstate = ys.YuMiState(armjnts)
            statelist.append(ajstate)
        et = time.time()
        print("time calculating sending information", et - st)
        # set the speed of the robot
        if speed_n == -1:
            armx.set_speed_max()
        else:
            self.rbtx.set_v(speed_n)
        exec_result = armx.movetstate_cont(statelist, is_add_all=self._is_add_all)
        return exec_result

    def calibrate_gripper(self):
        """
        Calibrate the gripper
        :param speed : float, optional
            Max speed of the gripper in mm/s.
            Defaults to 10 mm/s. If None, will use maximum speed in RAPID.
        :param force : float, optional
            Hold force used by the gripper in N.
            Defaults to 10 N. If None, will use maximum force the gripper can provide (20N).
        """
        self.rgt_arm_hnd.calibrate_gripper()
        self.lft_arm_hnd.calibrate_gripper()

    def __set_gripper_force(self, component_name: str, force: float = 10):
        """
        TODO: this program has bug. Fix it later.
        :param force: Hold force by the gripper in Newton.
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.set_gripper_force(force=force)

    def set_gripper_speed(self, component_name: str, speed: int = 10):
        """
        :param speed: In mm/s.
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.set_gripper_max_speed(max_speed=speed)

    def move_gripper(self, component_name: str, width: float):
        """
        Moves the gripper to the given width in meters.
        width : float
                Target width in meters, range[0 , 0.025]
                if you want to fully close or fully open the gripper,
                please use the open_gripper or close_gripper!!
                Otherwise the program may stuck
        """
        assert 0 <= width < yr.YMC.MAX_GRIPPER_WIDTH
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.move_gripper(width=width / 2)

    def open_gripper(self, component_name: str):
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.open_gripper()

    def close_gripper(self, component_name: str, force: float = 10):
        assert 0 <= force <= yr.YMC.MAX_GRIPPER_FORCE
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.close_gripper(force=force)

    def get_gripper_width(self, component_name: str):
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        return armx.get_gripper_width() * 2

    def _get_arm_jnts(self, armname: str):
        if armname == "rgt":
            return np.deg2rad(self.rbtx.right.get_state().joints)
        elif armname == "lft":
            return np.deg2rad(self.rbtx.left.get_state().joints)
        else:
            raise ValueError("Arm name must be right or left!")

    def get_hc_img(self, armname: str):
        if armname == "rgt":
            self.rbtx.right.write_handcamimg_ftp()
        elif armname == "lft":
            self.rbtx.left.write_handcamimg_ftp()
        else:
            raise ValueError("Arm name must be right or left!")

    def toggle_vac(self, toggletag, armname):
        if armname == "rgt":
            self.rbtx.right.toggle_vacuum(toggletag)
        elif armname == "lft":
            self.rbtx.left.toggle_vacuum(toggletag)

    def get_pressure(self, armname):
        if armname == "rgt":
            return self.rbtx.right.get_pressure()
        elif armname == "lft":
            return self.rbtx.left.get_pressure()

    def stop(self):
        self.rbtx.stop()


if __name__ == "__main__":
    ycc = Yumi_Controller(debug=False)
    ycc.set_gripper_speed("rgt_arm", 10)

    ycc.open_gripper("rgt_hnd")
    a = ycc.get_gripper_width("rgt_hnd")
    print(a)
    ycc.close_gripper("rgt_hnd", force=5)
    a = ycc.get_gripper_width("rgt_hnd")
    print(a)

    # ycc.get_pose("rgt_arm")
    # ycc.calibrate_gripper()
    # print(ycc.get_jnt_values("rgt_arm"))
    # ycc.set_gripper_speed("rgt_arm", 10)
