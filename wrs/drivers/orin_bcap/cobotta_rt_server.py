import copy
import socket
import struct
import time
import wrs.robot_con.cobotta.cobotta_x as cbtx
import numpy as np
import wrs._misc.promote_rt as pr


class CobottaRTServer(object):
    def __init__(self, pc_ip='192.168.0.2', port=18400):
        self._pc_server_socket_addr = (pc_ip, port)
        self._pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._pc_server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._pc_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._pc_server_socket.bind(self._pc_server_socket_addr)
        self._pc_server_socket.listen(5)
        self.path_list = []
        print(f"server open on {pc_ip}")
        self.robot_x = cbtx.CobottaX()
        pr.set_realtime()

    def move_jnts_motion(self, path):
        """
        move robot arm following a given path
        :param path: a list of 1x6 arrays
        author: junbo zhang
        date: 20211228
        """
        self.robot_x.move_jnts_motion(path)

    def move_jnts_motion_interpolated(self, path_interpolated):
        """
        move robot arm following a given interpolated path
        :param path_interpolated: a list of 1x6 arrays
        author: junbo zhang
        date: 20211228
        """
        self.robot_x.bcc.robot_execute(self.robot_x.hrbt, "slvChangeMode", 0x202)
        time.sleep(0.4)
        for jnt_values in path_interpolated:
            jnt_values_degree = np.degrees(jnt_values)
            self.robot_x.bcc.robot_execute(self.robot_x.hrbt, "slvMove", jnt_values_degree.tolist() + [0, 0])
        self.robot_x.bcc.robot_execute(self.robot_x.hrbt, "slvChangeMode", 0x000)
        time.sleep(0.2)

    def get_buffer(self):
        """
        receive buffer from client and execute the corresponding command
        author: junbo zhang
        date: 20211228
        """
        # accept pc socket
        pc_server_socket, pc_server_socket_addr = self._pc_server_socket.accept()
        print("PC server connected by ", pc_server_socket_addr)
        # receive buffer
        self.path_list = []
        self.buf_list = bytes()
        while True:
            buf = pc_server_socket.recv(1024)
            if buf == b"exit":
                print("connection break")
                break
            elif buf == b"joints":
                cur_jnts = self.robot_x.get_jnt_values()
                buf_jnts = bytes(struct.pack("!ffffff", cur_jnts[0], cur_jnts[1], cur_jnts[2],
                                             cur_jnts[3], cur_jnts[4], cur_jnts[5]))
                pc_server_socket.send(buf_jnts)
            elif buf == b"end_type":
                self.map_buffer_to_path()
            elif buf == b"run":
                path_execute = copy.deepcopy(self.path_list)
                self.path_list = []
                self.motion_execute(path_execute)
            elif len(buf) > 0:
                self.buf_list += buf
        pc_server_socket.close()

    def map_buffer_to_path(self):
        """
        convert buffer to path
        author: junbo zhang
        date: 20211228
        """
        print("buf axis_length:", len(self.buf_list))
        pose_num = int(len(self.buf_list) / 24)
        path = []
        for i in range(pose_num):
            pose = np.asarray(struct.unpack("!ffffff", self.buf_list[:24]))
            self.buf_list = self.buf_list[24:]
            path.append(pose)
        self.buf_list = bytes()
        print("path axis_length:", len(path))
        self.path_list.append(path)

    def motion_execute(self, path_list):
        """
        design the action sequence of the cobotta robot
        :param path_list: a list of paths for each stage
        author: junbo zhang
        date: 20211228
        """
        path_new_tip = path_list[1] + path_list[2]
        self.move_jnts_motion(path_list[0])
        self.robot_x.defult_gripper()
        self.move_jnts_motion(path_new_tip)
        self.move_jnts_motion(path_list[3])
        self.robot_x.open_gripper(speed=80)
        self.move_jnts_motion(path_list[4])
        self.robot_x.defult_gripper(speed=50)
        self.move_jnts_motion(path_list[5])
        self.move_jnts_motion(path_list[6])
        self.robot_x.open_gripper()
        self.robot_x.defult_gripper()
        self.move_jnts_motion(path_list[7])
        self.robot_x.close_gripper()
        self.robot_x.defult_gripper()


if __name__ == "__main__":
    rt_x = CobottaRTServer()
    num_client = 0
    while True:
        num_client += 1
        print(f"client{num_client}")
        rt_x.get_buffer()
