import socket
import struct
import wrs.motion.trajectory.piecewisepoly_toppra as trajp
import time
import math

import numpy as np


class CobottaRTClient(object):
    def __init__(self, server_ip="192.168.0.2", server_port=18400):
        self._pc_server_socket_addr = (server_ip, server_port)
        self._pc_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._pc_server_socket.connect(self._pc_server_socket_addr)
        self._jnts_scaler = 1e6
        self.traj_gen = trajp.PiecewisePolyTOPPRA()

    def send_jnts_motion_interpolated(self, path, toggle_debug=False):
        """
        send joints angles of interpolated path to server
         :param path: a list of 1x6 arrays
         :param toggle_debug:
        author: junbo zhang
        date: 20211228
        """
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        path = new_path
        vels_list = [math.pi * 2 / 3] * 6
        vels_list[1] = math.pi / 2
        interpolated_confs = self.traj_gen.interpolate_by_max_spdacc \
            (path, control_frequency=.008, max_vels=vels_list, toggle_debug=toggle_debug)
        self.send_jnts_motion(interpolated_confs)
        time.sleep(0.1)

    def send_jnts_motion(self, path):
        """
        send joints angles of path to server
         :param path: a list of 1x6 arrays
        author: junbo zhang
        date: 20211228
        """
        buf = bytes()
        for id, pose in enumerate(path):
            buf += struct.pack('!ffffff', pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
        print("buf axis_length:", len(buf))
        self._pc_server_socket.send(buf)
        time.sleep(0.3)
        self._pc_server_socket.send(struct.pack("!3s", b"end_type"))

    def start_execution(self):
        """
        send ending signal to server
        author: junbo zhang
        date: 20211228
        """
        self._pc_server_socket.send(struct.pack("!3s", b"run"))

    def get_jnts_value(self):
        """
        send request to server, and get current joints values from server
        :return: 1x6 arrays
        author: junbo zhang
        date: 20211228
        """
        self._pc_server_socket.send(struct.pack("!4s", b"joints"))
        jnts_buf = self._pc_server_socket.recv(1024)
        return np.asarray(struct.unpack("!ffffff", jnts_buf))

    def close_connection(self):
        """
        send closing connection signal to server
        author: junbo zhang
        date: 20211228
        """
        self._pc_server_socket.send(struct.pack("!4s", b"exit"))