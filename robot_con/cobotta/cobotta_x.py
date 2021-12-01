import motion.trajectory.piecewisepoly_toppra as trajp
import drivers.orin_bcap.bcapclient as bcapclient


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
        self.traj_gen = trajp.PiecewisePolyTOPPRA()

    def __del__(self):
        self.clear_error()
        self.bcc.controller_getrobot(self.hrbt, "Motor", [0, 0])
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        self.bcc.robot_release(self.hrbt)
        self.bcc.controller_disconnect(self.hctrl)
        self.bcc.service_stop()

    def clear_error(self):
        self.bcc.controller_execute(self.hctrl, "ClearError", None)

    def get_jnt_values(self):
        pose = self.bcc.robot_execute(self.hrbt, "CurJnt", None)
        return np.radians(np.array(pose[:6]))

    def move_jnts(self, jnt_values):
        """
        :param jnt_values:  1x6 np array
        :return:
        author: weiwei
        date: 20210507
        """
        jnt_values_degree = np.degrees(jnt_values)
        self.bcc.robot_move(self.hrbt, 1, [jnt_values_degree.tolist(), "J", "@E"], "")

    def move_jnts_motion(self, path, toggle_debug=False):
        """
        :param path:
        :return:
        author: weiwei
        date: 20210507
        """
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        path = new_path
        interpolated_confs = \
            self.traj_gen.interpolate_by_max_spdacc(path, control_frequency=.008, toggle_debug=toggle_debug)
        # Slave move: Change mode
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x202)
        for jnt_values in interpolated_confs:
            jnt_values_degree = np.degrees(jnt_values)
            self.bcc.robot_execute(self.hrbt, "slvMove", jnt_values_degree.tolist() + [0, 0])
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x000)

    def move_jnts(self, jnt_values):
        """
        :param jnt_values:  1x6 np array
        :return:
        author: weiwei
        date: 20210507
        """
        jnt_values_degree = np.degrees(jnt_values)
        self.bcc.robot_move(self.hrbt, 1, [jnt_values_degree.tolist(), "J", "@E"], "")

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
    import basis.robot_math as rm
    import robot_sim.robots.cobotta.cobotta as cbt
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

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
