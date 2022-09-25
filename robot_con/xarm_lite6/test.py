import time
import unittest
import logging

import numpy as np

from xarm_lite6_x import XArmLite6X


class TestXArmLite6X(unittest.TestCase):
    def setUp(self):
        self.armx = XArmLite6X()

    def test_get_pose(self):
        pose = self.armx.get_pose()
        print("Current position is", pose[0])
        print("Current orientation is", pose[1])

    def test_get_jnt_values(self):
        jnts = self.armx.get_jnt_values()
        print("Current joint values are", jnts)

    def test_move_p(self):
        suc = self.armx.move_p(pos=np.array([0.3, 0.02, 0.3]), rot=np.array([3.1415926, 0, 0]), speed=1000)
        print("Is success", suc)
        pose = self.armx.get_pose()
        print("Current position is", pose[0])
        print("Current orientation is", pose[1])
        print("Current joint values are", self.armx.get_jnt_values())

        self.assertTrue(suc)

    def test_move_j(self):
        suc = self.armx.move_j(jnt_val=np.array([0.049958, -0.213357, 1.085769, 0., 1.299126, 0.049958]),
                               speed=200)
        print("Is success", suc)
        pose = self.armx.get_pose()
        print("Current position is", pose[0])
        print("Current orientation is", pose[1])
        print("Current joint values are", self.armx.get_jnt_values())
        self.assertTrue(suc)

    def test_ik(self):
        st = time.time()
        iks = self.armx.ik(tgt_pos=np.array([0.2, 0.1, 0.3]), tgt_rot=np.array([3.1415926, 0, 0]), )
        et = time.time()
        print(f"IK solution is {repr(iks)}. Time consuming is {1000000 * (et - st)} us")

    def test_reset(self):
        self.armx.reset()

    def test_home(self):
        self.armx.homeconf()


if __name__ == '__main__':
    unittest.main()
