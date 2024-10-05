import math
import time
import threading
import socket
import struct
import os
import wrs.basis.robot_math as rm
import wrs.drivers.urx.ur_robot as urrobot
import wrs.motion.trajectory.topp_ra as pwp
import wrs.robot_con.ur.program_builder as pb
from wrs.robot_con.ur.robotiq import rtq_cbseries_gripper as r2f
from wrs.robot_con.ur.robotiq import rtq_ft300 as rft


class UR3Rtq85X(object):
    """
    author: weiwei
    date: 20180131
    """

    def __init__(self, robot_ip='10.2.0.50', pc_ip='10.2.0.100'):
        """
        :param robot_ip:
        :param pc_ip:
        """
        # setup arm
        self._arm = urrobot.URRobot(robot_ip)
        self._arm.set_tcp((0, 0, 0, 0, 0, 0))
        self._arm.set_payload(1.28)
        # setup hand
        self._hnd = r2f.RobotiqCBTwoFinger(type='rtq85')
        # setup ftsensor
        self._ftsensor = rft.RobotiqFT300()
        self._ftsensor_socket_addr = (robot_ip, 63351)
        self._ftsensor_urscript = self._ftsensor.get_program_to_run()
        # setup pc server
        self._pc_server_socket_addr = (pc_ip, 0)  # 0: the system finds an available port
        self._pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._pc_server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._pc_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pc_server_socket.bind(self._pc_server_socket_addr)
        self._pc_server_socket.listen(5)
        self._jnts_scaler = 1e6
        self._pb = pb.ProgramBuilder()
        self._script_dir = os.path.dirname(__file__)
        self._pb.load_prog(os.path.join(self._script_dir, "urscripts_cbseries/moderndriver_cbseries.script"))
        self._modern_driver_urscript = self._pb.get_program_to_run()
        self._modern_driver_urscript = self._modern_driver_urscript.replace("parameter_ip",
                                                                            self._pc_server_socket_addr[0])
        self._modern_driver_urscript = self._modern_driver_urscript.replace("parameter_port",
                                                                            str(self._pc_server_socket.getsockname()[
                                                                                    1]))
        self._modern_driver_urscript = self._modern_driver_urscript.replace("parameter_jointscaler",
                                                                            str(self._jnts_scaler))
        self._ftsensor_thread = None
        self._ftsensor_values = []
        self.trajt = pwp

    @property
    def arm(self):
        # read-only property
        return self._arm

    @property
    def hnd(self):
        # read-only property
        return self._hnd

    @property
    def ftsensor_urscript(self):
        # read-only property
        return self._ftsensor_urscript

    @property
    def ftsensor_socket_addr(self):
        # read-only property
        return self._ftsensor_socket_addr

    @property
    def pc_server_socket_info(self):
        """
        :return: [ip, port]
        """
        return self._pc_server_socket.getsockname()

    @property
    def pc_server_socket(self):
        return self._pc_server_socket

    @property
    def jnts_scaler(self):
        return self._jnts_scaler

    def open_gripper(self, speed_percentage=70, force_percentage=50, finger_distance=85):
        """
        open the rtq85 hand on the arm specified by arm_name
        :param arm_name:
        :return:
        author: weiwei
        date: 20180220
        """
        self._arm.send_program(
            self._hnd.get_actuation_program(speed_percentage, force_percentage, finger_distance=finger_distance))
        time.sleep(.2)
        while self._arm.is_program_running():
            time.sleep(.01)

    def close_gripper(self, speed_percentage=80, force_percentage=50):
        """
        close the rtq85 hand on the arm specified by arm_name
        :param arm_name:
        :return:
        author: weiwei
        date: 20180220
        """
        self._arm.send_program(self._hnd.get_actuation_program(speed_percentage, force_percentage, finger_distance=0))
        time.sleep(.2)
        while self._arm.is_program_running():
            time.sleep(.01)

    def start_recvft(self):
        """
        start receive ft values using thread
        the values are in the local frame of the force sensors
        transformation is to be done by higher-level code
        :return: [fx, fy, fz, tx, ty, tz] in N and Nm
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

    def move_jnts(self, jnt_values, wait=True):
        """
        :param jnt_values: a 1-by-6 list in degree
        :param arm_name:
        :return:
        author: weiwei
        date: 20170411
        """
        self._arm.movej(jnt_values, acc=1, vel=1, wait=wait)
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

    def move_jspace_path(self, path, ctrl_freq=.008, max_vels=None, max_accs=None):
        """
        move robot_s arm following a given jointspace path
        :param path: a list of 1x6 arrays
        :param ctrl_freq:
        :param max_vels
        :param max_accs
        :return:
        author: weiwei
        date: 20210331
        """
        _, interp_confs, _, _ = self.trajt.generate_time_optimal_trajectory(path,
                                                                            max_vels=max_vels,
                                                                            max_accs=max_accs,
                                                                            ctrl_freq=ctrl_freq)
        # upload a urscript to connect to the pc server started by this class
        self._arm.send_program(self._modern_driver_urscript)
        # accept arm socket
        rbt_socket, rbt_socket_addr = self._pc_server_socket.accept()
        print("PC server onnected by ", rbt_socket_addr)
        # send trajectory
        keepalive = 1
        buf = bytes()
        for id, conf in enumerate(interp_confs):
            if id == len(interp_confs) - 1:
                keepalive = 0
            jointsradint = [int(jnt_value * self._jnts_scaler) for jnt_value in conf]
            buf += struct.pack('!iiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                               jointsradint[3], jointsradint[4], jointsradint[5], keepalive)
        rbt_socket.send(buf)
        rbt_socket.close()

    def get_jnt_values(self):
        """
        get the joint angles in radian
        :param arm_name:
        :return:
        author: ochi, revised by weiwei
        date: 20180410
        """
        return self._arm.getj()

    def get_jaw_width(self):
        self._arm.send_program(self._hnd.get_jaw_width_program(self.pc_server_socket))
        rbt_socket, rbt_socket_addr = self.pc_server_socket.accept()
        value = rbt_socket.recv(1024).decode(encoding='ascii')
        rbt_socket.close()
        return float(value)*.001


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 1, 2], lookat_pos=[0, 0, 0])
    u3r85_x = UR3Rtq85X(robot_ip='10.2.0.50', pc_ip='10.2.0.100')
    u3r85_x.open_gripper(finger_distance=30)
    print(u3r85_x.get_jaw_width())
    print(u3r85_x.get_jaw_width())
    print(u3r85_x.get_jnt_values())
    base.run()
