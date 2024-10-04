import copy
import time
import wrs.motion.trajectory.piecewisepoly_toppra as trajp
from wrs import drivers as bcapclient
import numpy.typing as npt
from typing import List


class CobottaX(object):

    def __init__(self, host='192.168.0.1', port=5007, timeout=2000):
        """
        :param host:
        :param port:
        :param timeout:

        author: weiwei
        date: 20210507
        """
        self.bcc = bcapclient.BCAPClient(host, port, timeout)
        self.bcc.service_start("")
        # Connect to RC8 (RC8(VRC)provider)
        self.hctrl = self.bcc.controller_connect("", "CaoProv.DENSO.VRC", ("localhost"), (""))
        self.clear_error()
        # get robot_s object hanlde
        self.hrbt = self.bcc.controller_getrobot(self.hctrl, "Arm", "")
        # print(self.bcc.robot_getvariablenames(self.hrbt))
        # self.bcc.controller_getextension(self.hctrl, "Hand", "")
        # take arm
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        # motor on
        self.bcc.robot_execute(self.hrbt, "Motor", [1, 0])
        # set ExtSpeed = [speed, acc, dec]
        self.bcc.robot_execute(self.hrbt, "ExtSpeed", [100, 100, 100])
        self.traj_gen = trajp.PiecewisePolyTOPPRA()
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,1,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,2,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,3,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,4,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,5,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,6,0.01])

    def __del__(self):
        self.clear_error()
        self.bcc.controller_getrobot(self.hrbt, "Motor", [0, 0])
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        self.bcc.robot_release(self.hrbt)
        self.bcc.controller_disconnect(self.hctrl)
        self.bcc.service_stop()

    def clear_error(self):
        self.bcc.controller_execute(self.hctrl, "ClearError", None)

    def disconnect(self):
        self.bcc.controller_disconnect(self.hctrl)

    def moveto_named_pose(self,name):
        self.bcc.robot_move(self.hrbt,1,name,"")

    def move_jnts_motion(self, path: List[npt.NDArray[float]], toggle_debug: bool = False):
        """
        :param path:
        :return:
        author: weiwei
        date: 20210507
        """
        time.sleep(0.1)
        try:
            self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])  # 20220317, needs further check, speedmode?
        except:
            self.clear_error()
            time.sleep(0.2)
            self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.2)
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        path = new_path
        max_vels = [math.pi * .3, math.pi * .2, math.pi/2, math.pi/2, math.pi/2, math.pi * 0.75]
        interpolated_confs = \
            self.traj_gen.interpolate_by_max_spdacc(path,
                                                    control_frequency=.008,
                                                    max_vels=max_vels,
                                                    toggle_debug=toggle_debug)
        # print(f"toppra{interpolated_confs[:,2].max()}")
        # Slave move: Change mode
        while True:
            try:
                # time.sleep(.2)
                self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x202)
                time.sleep(.5)
                # print("sleep done")
                print(self.get_jnt_values())
                print(interpolated_confs[0].tolist())
                self.bcc.robot_execute(self.hrbt, "slvMove", np.degrees(interpolated_confs[0]).tolist() + [0, 0])
                # time.sleep(.2)
                # print("try exec done")
                break
            except:
                # print("exception, continue")
                self.clear_error()
                time.sleep(0.2)
                continue
        try:
            for jnt_values in interpolated_confs:
                jnt_values_degree = np.degrees(jnt_values)
                self.bcc.robot_execute(self.hrbt, "slvMove", jnt_values_degree.tolist() + [0, 0])
            # print("trajectory done")
        except:
            # print("trajectory exception, continue")
            self.clear_error()
            time.sleep(0.2)
            return False
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x000)
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)
        return True

    def get_jnt_values(self):
        pose = self.bcc.robot_execute(self.hrbt, "CurJnt", None)
        return np.radians(np.array(pose[:6]))

    def get_pose_values(self):
        """
        x,y,z,r,p,y,fig
        :return:
        author: weiwei
        date: 20220115
        """
        pose = self.bcc.robot_execute(self.hrbt, "CurPos", None)
        return_value = np.array(pose[:7])
        return_value[:3] *= .001
        return_value[3:6] = np.radians(return_value[3:6])
        return return_value

    def move_jnts(self, jnt_values: npt.NDArray[float]):
        """
        :param jnt_values:  1x6 np array
        :return:
        author: weiwei
        date: 20210507
        """
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.1)
        jnt_values_degree = np.degrees(jnt_values)
        self.bcc.robot_move(self.hrbt, 1, [jnt_values_degree.tolist(), "J", "@E"], "")
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)

    def move_pose(self, pose, speed=100):
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.1)
        pose = np.array(pose)
        pose_value = copy.deepcopy(pose)
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        self.bcc.robot_move(self.hrbt, 1, [pose_value.tolist(), "P", "@E"], f"SPEED={speed}")
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)

    def open_gripper(self, dist=.021, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def close_gripper(self, dist=.0, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def defult_gripper(self, dist=.014, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def P2J(self, pose):
        pose = np.array(pose)
        pose_value = copy.deepcopy(pose)
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        return np.radians(self.bcc.robot_execute(self.hrbt, "P2J", pose_value.tolist()))[:6]

    def J2P(self, jnt_values):
        jnt_values = np.array(jnt_values)
        jnt_values_degree = np.degrees(jnt_values)
        pose_value = np.radians(self.bcc.robot_execute(self.hrbt, "J2P", jnt_values_degree.tolist()))
        return_value = np.array(pose_value[:7])
        return_value[:3] *= .001
        return_value[3:6] = np.radians(return_value[3:6])
        return return_value

    def null_space_search(self, current_pose):
        pose = copy.deepcopy(current_pose)
        times = 0
        for angle in range(0, 180,5):
            for i in [-1, 1]:
                try:
                    self.P2J(pose)
                    return pose, times
                except:
                    self.clear_error()
                    times += 1
                    time.sleep(0.1)
                    pose[5] = current_pose[5] + np.radians(angle * i)
        return None, times


if __name__ == '__main__':
    import math
    import numpy as np

    # base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    # mgm.gen_frame().attach_to(base)

    # robot_s = cbt.CobottaRIPPS()
    robot_x = CobottaX()
    # for i in range(5):
    #     print(i)
    #     robot_x = CobottaX()
    #     robot_x.disconnect()
    # robot_x.defult_gripper()
    print(robot_x.get_pose_values())
    print(robot_x.get_jnt_values())

    eject_jnt_values1 = np.array([1.37435462 , 0.98535585,  0.915062 ,   1.71130978, -1.23317083 ,-0.93993529])
    eject_jnt_values2 = np.array([1.21330229,  1.04185468,  0.84606752,  1.61686219, -1.11360179, -0.90834774])
    eject_jnt_values3 = np.array([1.41503157 , 0.67831314 , 1.41330569 , 1.45821765, -1.41208026 ,-0.45279814])
    eject_jnt_values4 = np.array([1.16644694,  0.883306 ,   1.07903103,  1.47912526, -1.41216692, -0.64314669])
    eject_jnt_values_list = [eject_jnt_values1, eject_jnt_values2, eject_jnt_values3, eject_jnt_values4]
    record_poae = np.array([1.02445228e-01,  1.48697438e-01,  2.60223845e-01 , 1.57047373e+00, -6.51089430e-04,  1.57065059e+00,  5.00000000e+00])

    def move_to_new_pose(pose, speed=100):
        pose, times = robot_x.null_space_search(pose)
        if pose is not None:
            robot_x.move_pose(pose, speed=speed)
            return times
        else:
            raise Exception("No solution!")

    current_jnts=robot_x.get_jnt_values()
    print("pose:",robot_x.get_pose_values())
    plant_pose = np.array([0.2270, 0.2405, 0.2045, np.pi / 2, 0, np.pi / 2, 5])
    plant_pose_next = np.array([0.1170, 0.2405, 0.2045, np.pi / 2, 0, np.pi / 2, 5])
    robot_x.open_gripper(speed=70)
    robot_x.defult_gripper()
    robot_x.open_gripper(speed=100)
    robot_x.defult_gripper()

    # detect_pose = copy.deepcopy(plant_pose)
    # detect_pose[0] -= 0.025
    # robot_x.move_pose(detect_pose)
    #
    # time.sleep(1)
    #
    # detect_pose_next = copy.deepcopy(plant_pose_next)
    # detect_pose_next[0] -= 0.025
    # robot_x.move_pose(detect_pose_next)
