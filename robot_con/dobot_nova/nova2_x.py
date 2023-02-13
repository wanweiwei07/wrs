"""
WRS control interface for Nova 2
Author: Hao Chen <chen960216@gmail.com>, 20230113, osaka
Update Note <Version:0.0.1,20230113>: Create a basic API for the WRS system
"""
from typing import Optional
import numpy as np
import basis.robot_math as rm
from drivers.dobot_tcp import Dobot

try:
    import motion.trajectory.piecewisepoly_toppra as pwp
    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

__VERSION__ = (0, 0, 1)


class Nova2X(object):
    @staticmethod
    def pos_arm2wrs(arr: np.ndarray) -> np.ndarray:
        """
        Convert the position in Arm API to the WRS system
        :param arr: Position array obtained from the XArm API
        :return: Converted position array
        """
        return arr / 1000

    @staticmethod
    def pos_wrs2arm(arr: np.ndarray) -> np.ndarray:
        """
        Convert the position in WRS system to the Arm API
        :param arr: Position array in the WRS system
        :return: Converted position array
        """
        return arr * 1000

    @staticmethod
    def angle_arm2wrs(arr: np.ndarray) -> np.ndarray:
        """
        Convert the angle in Arm API to the WRS system
        :param arr: angle array obtained from the Arm API in degree
        :return: Converted angle array in radian
        """
        return np.deg2rad(arr)

    @staticmethod
    def angle_wrs2arm(arr: np.ndarray) -> np.ndarray:
        """
        Convert the angle in WRS system to the Arm API
        :param arr: Position array in the WRS system
        :return: Converted position array
        """
        return np.rad2deg(arr)

    def __init__(self, ip: str = "192.168.5.1"):
        """
        :param ip: The ip address of the robot
        """
        # examine parameters
        # initialization
        self._arm_x = Dobot(ip=ip)
        self._arm_x.clear_error()
        if not self._arm_x.is_enable:
            self._arm_x.power_on()
            self._arm_x.enable_robot()
        self.ndof = 6

    @property
    def mode(self) -> int:
        """
        Dobot mode
        :return:   If the brake is released, the mode is 2.
                   If the robot is powered on but not enabled, the mode is 4.
                   If the robot is enabled successfully, the mode is 5.
                   If the robot runs, the mode is 7.
                   If the robot pauses, the mode is 10.
                   If the robot enters drag mode (enabled state), the mode is 6.
                   If the robot is dragging and recording, the mode is 8.
                   If the robot is jogging, the mode is 11.
                   Alarm is the top priority. When other modes exist simultaneously, if there is an alarm, the mode is set to 9 first.

        """
        return self._arm_x.robot_mode

    def clear_error(self):
        self._arm_x.clear_error()

    def reset(self):
        self._arm_x.reset_robot()

    def ik(self, tgt_pos: np.ndarray, tgt_rot: np.ndarray, seed_jnts: np.ndarray = None) -> Optional[np.ndarray]:
        """

        :param tgt_pos: The position under WRS system
        :param tgt_rot: The 3x3 Rotation matrix or 1x3 RPY matrix
        :return: inverse kinematics solution
        """
        if tgt_rot.shape == (3, 3):
            tgt_rpy = rm.rotmat_to_euler(tgt_rot)
        else:
            tgt_rpy = tgt_rot.flatten()[:3]
        ik_sol = self._arm_x.ik(pos=self.pos_wrs2arm(tgt_pos),
                                rot=self.angle_wrs2arm(tgt_rpy),
                                seed_jnts=self.angle_wrs2arm(seed_jnts))
        if ik_sol is None:
            return
        return self.angle_arm2wrs(ik_sol)

    def get_jnt_values(self) -> np.ndarray:
        """
        Get the joint values of the arm
        :return: Joint values (Array)
        """
        return self.angle_arm2wrs(self._arm_x.get_jnt())

    def get_pose(self) -> (np.ndarray, np.ndarray):
        """
        Get the cartesian position
        :return: tuple(Position(Array), Orientation(Array))
        """
        pose = self._arm_x.get_tcp_cartesian()
        return self.pos_arm2wrs(np.array(pose[:3])), rm.rotmat_from_euler(*self.angle_arm2wrs(pose[3:]))

    def move_j(self, jnt_val: np.ndarray, ):
        """
        Move the robot to a target joint value
        :param jnt_val: Targe joint value (1x6 Array)
        :return: if the path is moved successfully, it will return 0
        """
        assert isinstance(jnt_val, np.ndarray) and len(jnt_val) == self.ndof
        self._arm_x.movej(self.angle_wrs2arm(jnt_val))

    def move_p(self, pos: np.ndarray, rot: np.ndarray, is_linear: bool = True):
        """
        Move to a pose under the robot base coordinate
        :param pos: Position (Array([x,y,z])) of the pose
        :param rot: Orientation (Array([roll,pitch,yaw]) or Array(3x3)) of the pose
        :param is_linear: bool, if True is linear movement
        """
        pos = self.pos_wrs2arm(pos)
        rot = np.array(rot)
        if rot.shape == (3, 3):
            rpy = rm.rotmat_to_euler(rot)
        else:
            rpy = rot.flatten()[:3]
        rpy = self.angle_wrs2arm(rpy)
        if is_linear:
            self._arm_x.movel(pos, rpy)
        else:
            self._arm_x.movep(pos, rpy)

    def move_jntspace_path(self, path,
                           max_jntvel: list = None,
                           max_jntacc: list = None,
                           start_frame_id=1,
                           toggle_debug=False):
        raise NotImplementedError

    def __del__(self):
        self._arm_x.close()


if __name__ == "__main__":
    rbtx = Nova2X()
    print(f"Mode is {rbtx.mode}")
    print(repr(rbtx.get_jnt_values()))
    print(rbtx.get_pose())
    rbtx.move_p(np.array([0.20320204, -0.25951422, 0.30715834]),
                np.array([[-0.08450983, -0.8993117, -0.42906474],
                          [-0.99284347, 0.03953477, 0.11268916],
                          [-0.0843797, 0.43551747, -0.89621683]]))
