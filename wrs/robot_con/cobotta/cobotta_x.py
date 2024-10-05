import time
import numpy.typing as npt
from typing import List
import wrs.motion.trajectory.topp_ra as trajp


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
        # self.bcc.controller_getextension(self.hctrl, "Hand", "")
        # take arm
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        # motor on
        self.bcc.robot_execute(self.hrbt, "Motor", [1, 0])
        # set ExtSpeed = [speed, acc, dec]
        self.bcc.robot_execute(self.hrbt, "ExtSpeed", [100, 100, 100])

    def __del__(self):
        self.clear_error()
        self.bcc.controller_getrobot(self.hrbt, "Motor", [0, 0])
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        self.bcc.robot_release(self.hrbt)
        self.bcc.controller_disconnect(self.hctrl)
        self.bcc.service_stop()

    def clear_error(self):
        self.bcc.controller_execute(self.hctrl, "ClearError", None)

    def move_jnts_motion(self, path: List[npt.NDArray[float]], toggle_debug: bool = False):
        """
        :param path:
        :return:
        author: weiwei
        date: 20210507
        """
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])  # 20220319 robot_move changed speed limits?
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        path = new_path
        max_vels = [math.pi * .6, math.pi * .4, math.pi, math.pi, math.pi, math.pi * 1.5]
        interpolated_confs = \
            trajp.generate_time_optimal_trajectory(path,
                                                   max_vels=max_vels,
                                                   ctrl_freq=.008)
        # Slave move: Change mode
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x202)
        time.sleep(.02)
        for jnt_values in interpolated_confs:
            jnt_values_degree = np.degrees(jnt_values)
            self.bcc.robot_execute(self.hrbt, "slvMove", jnt_values_degree.tolist() + [0, 0])
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x000)

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
        jnt_values_degree = np.degrees(jnt_values)
        self.bcc.robot_move(self.hrbt, 1, [jnt_values_degree.tolist(), "J", "@E"], "")

    def move_pose(self, pose_value):
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        self.bcc.robot_move(self.hrbt, 1, [pose_value.tolist(), "P", "@E"], "")

    def open_gripper(self, dist=.03):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, 100])

    def close_gripper(self, dist=.0):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, 100])


if __name__ == '__main__':
    import math
    import numpy as np
    from wrs import basis as rm, drivers as bcapclient, robot_sim as cbt, motion as rrtc, modeling as gm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)

    robot_s = cbt.Cobotta()
    robot_x = CobottaX()
    start_conf = robot_x.get_jnt_values()
    print("start_radians", start_conf)
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    rrtc_planner = rrtc.RRTConnect(robot_s)
    path = rrtc_planner.plan(component_name="arm",
                             start_conf=start_conf,
                             goal_conf=jnt_values,
                             ext_dist=.1,
                             max_time=300)
    robot_x.move_jnts_motion(path)
    robot_x.close_gripper()
    for pose in path:
        robot_s.fk("arm", pose)
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
    base.run()
