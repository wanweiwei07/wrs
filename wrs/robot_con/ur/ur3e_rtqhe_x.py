import math
import numpy as np
import threading
import socket
import struct
import os
import wrs.basis.robot_math as rm
import wrs.drivers.urx.ur_robot as urrobot
import wrs.motion.trajectory.piecewisepoly_toppra as pwp
import wrs.robot_con.ur.program_builder as pb
from wrs.robot_con.ur.robotiq import rtq_eseries_gripper as r2f


class UR3ERtqHEX():
    """
    author: weiwei
    date: 20180131, 20210401osaka
    """

    def __init__(self, robot_ip='10.2.0.50', pc_ip='10.2.0.91'):
        """
        :param robot_ip:
        :param pc_ip:
        """
        # setup arm
        self._arm = urrobot.URRobot(robot_ip)
        self._arm.set_tcp((0, 0, 0, 0, 0, 0))
        self._arm.set_payload(1.0)
        # setup hand
        self._hand = r2f.RobotiqETwoFinger(type='hande')
        # # setup ftsensor
        # self._ftsensor = rft.RobotiqFT300()
        # self._ftsensor_socket_addr = (robot_ip, 63351)
        # self._ftsensor_urscript = self._ftsensor.get_program_to_run()
        # setup pc server
        self._pc_server_socket_addr = (pc_ip, 0)  # 0: the system finds an available port
        self._pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._pc_server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._pc_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pc_server_socket.bind(self._pc_server_socket_addr)
        self._pc_server_socket.listen(5)
        self._jointscaler = 1e6
        self._pb = pb.ProgramBuilder()
        script_dir = os.path.dirname(__file__)
        self._pb.load_prog(os.path.join(script_dir, "urscripts_eseries/moderndriver_eseries.script"))
        self._pc_server_urscript = self._pb.get_program_to_run()
        self._pc_server_urscript = self._pc_server_urscript.replace("parameter_ip", self._pc_server_socket_addr[0])
        self._pc_server_urscript = self._pc_server_urscript.replace("parameter_port",
                                                                    str(self._pc_server_socket.getsockname()[1]))
        self._pc_server_urscript = self._pc_server_urscript.replace("parameter_jointscaler",
                                                                    str(self._jointscaler))
        self._ftsensor_thread = None
        self._ftsensor_values = []
        self.tpply = pwp.PiecewisePolyTOPPRA()

    @property
    def arm(self):
        # read-only property
        return self._arm

    @property
    def ftsensor_urscript(self):
        # read-only property
        return self._ftsensor_urscript

    @property
    def ftsensor_socket_addr(self):
        # read-only property
        return self._ftsensor_socket_addr

    def open_gripper(self, speedpercentange=70, forcepercentage=50, fingerdistance=50.0):
        """
        open the rtq85 hand on the arm specified by arm_name
        :param arm_name:
        :return:
        author: weiwei
        date: 20180220
        """
        self._arm.send_program(self._hand.return_program_to_run(speedpercentange, forcepercentage, fingerdistance))

    def close_gripper(self, speed_percentange=80, force_percentage=100):
        """
        close the rtq85 hand on the arm specified by arm_name
        :param arm_name:
        :return:
        author: weiwei
        date: 20180220
        """
        self._arm.send_program(self._hand.return_program_to_run(speed_percentange, force_percentage, 0))

    def start_recvft(self):
        """
        start receive ft values using thread
        the values are in the local frame of the force sensors
        transformation is to be done by higher-level code
        :return:
        """

        def recvft():
            self._arm.send_program(self._ftsensor_urscript)
            ftsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ftsocket.connect(self._ftsensor_socket_addr)
            while True:
                ftdata = ftsocket.recv(1024)
                ftdata = ftdata.decode()
                ftdata = ftdata.strip('()')
                self._ftsensor_values.append([float(x) for x in ftdata.split(',')])

        self._ftsensor_thread = threading.Thread(target=recvft, name="threadft")
        self._ftsensor_thread.start()

    def stop_recvft(self):
        self._ftsensor_thread.join()

    def reset_ftsensor(self):
        pass

    def clear_ftsensor_values(self):
        self._ftsensor_values = []

    def move_jnts(self, jnt_values, radius=0.01):
        """
        :param jnt_values: a 1-by-6 list in degree
        :param arm_name:
        :return:
        author: weiwei
        date: 20170411
        """
        jointsrad = [math.radians(angdeg) for angdeg in jnt_values]
        self._arm.movej(jointsrad, acc=1, vel=1, wait=True)
        # targetarm.movejr(jointsrad, acc = 1, vel = 1, major_radius = major_radius, wait = False)

    def regulate_jnts_pmpi(self):
        """
        TODO allow settings for pmpi pm 2pi
        the function move all joints back to -360,360
        due to improper operations, some joints could be out of 360
        this function moves the outlier joints back
        :return:
        author: weiwei
        date: 20180202
        """
        jnt_values = self.get_jnt_values()
        regulated_jnt_values = rm.regulate_angle(-math.pi, math.pi, jnt_values)
        self.move_jnts(regulated_jnt_values)

    def move_jntspace_path(self, path, control_frequency=.005):
        """
        move robot_s arm following a given jointspace path
        :param path:
        :param control_frequency: the program will sample time_intervals/control_frequency confs, see motion.trajectory
        :param interval_time: equals to expandis/speed, speed = degree/second
                              by default, the value is 1.0 and the speed is expandis/second
        :return:
        author: weiwei
        date: 20210331
        """
        max_jnt_vels = np.array([np.pi*.75, np.pi*.75, np.pi*.75, np.pi * 1.5, np.pi * 1.5, np.pi * 1.5])
        max_jnt_accs = np.array([np.pi*.75, np.pi*.75, np.pi*.75, np.pi * 1.5, np.pi * 1.5, np.pi * 1.5])
        interpolated_confs = self.tpply.interpolate_by_max_spdacc(path=path,
                                                                  control_frequency=control_frequency,
                                                                  max_vels=max_jnt_vels,
                                                                  max_accs=max_jnt_accs,
                                                                  toggle_debug=False)
        # upload a urscript to connect to the pc server started by this class
        self._arm.send_program(self._pc_server_urscript)
        # accept arm socket
        pc_server_socket, pc_server_socket_addr = self._pc_server_socket.accept()
        print("Connected by ", pc_server_socket_addr)
        # send trajectory
        keepalive = 1
        buf = bytes()
        for id, conf in enumerate(interpolated_confs):
            if id == len(interpolated_confs) - 1:
                keepalive = 0
            jointsradint = [int(jnt_value * self._jointscaler) for jnt_value in conf]
            buf += struct.pack('!iiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                               jointsradint[3], jointsradint[4], jointsradint[5], keepalive)
        pc_server_socket.send(buf)
        pc_server_socket.close()

    def get_jnt_values(self):
        """
        get the joint angles in radian
        :param arm_name:
        :return:
        author: ochi, revised by weiwei
        date: 20180410
        """
        return self._arm.getj()


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    u3erhe_x = UR3ERtqHEX(robot_ip='10.0.2.2', pc_ip='10.0.2.17')
    u3erhe_x.close_gripper()
    base.run()
