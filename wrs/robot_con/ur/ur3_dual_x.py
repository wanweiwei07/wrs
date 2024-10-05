import struct
import os
import numpy as np
import wrs.motion.trajectory.topp_ra as trajp
import wrs.robot_con.ur.ur3_rtq85_x as u3r85x
import wrs.robot_con.ur.program_builder as pb


class UR3DualX(object):
    """
    urx 50, right arm 51, left arm 52
    author: weiwei
    date: 20180131
    """

    def __init__(self, lft_robot_ip='10.2.0.50', rgt_robot_ip='10.2.0.51', pc_ip='10.2.0.100'):
        """
        :param robot_sim: for global transformation, especially in attachfirm
        author: weiwei
        date: 20191014 osaka
        """
        self._lft_arm = u3r85x.UR3Rtq85X(robot_ip=lft_robot_ip, pc_ip=pc_ip)
        self._rgt_arm = u3r85x.UR3Rtq85X(robot_ip=rgt_robot_ip, pc_ip=pc_ip)
        self._pb = pb.ProgramBuilder()
        current_file_dir = os.path.dirname(__file__)
        self._pb.load_prog(os.path.join(current_file_dir, "urscripts_cbseries/moderndriver_cbseries_master.script"))
        self._master_modern_driver_urscript = self._pb.get_program_to_run()
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_pc_ip",
                                                                                          self._lft_arm.pc_server_socket_info[
                                                                                              0])
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_pc_port",
                                                                                          str(
                                                                                              self._lft_arm.pc_server_socket_info[
                                                                                                  1]))
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_slave_ip",
                                                                                          rgt_robot_ip)
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_jnts_scaler",
                                                                                          str(self._lft_arm.jnts_scaler))
        self._pb.load_prog(os.path.join(current_file_dir, "urscripts_cbseries/moderndriver_cbseries_slave.script"))
        self._slave_modern_driver_urscript = self._pb.get_program_to_run()
        self._slave_modern_driver_urscript = self._slave_modern_driver_urscript.replace("parameter_master_ip",
                                                                                        lft_robot_ip)
        self._slave_modern_driver_urscript = self._slave_modern_driver_urscript.replace("parameter_jnts_scaler",
                                                                                        str(self._lft_arm.jnts_scaler))
        # print(self._slave_modern_driver_urscript)

    @property
    def lft_arm(self):
        # read-only property
        return self._lft_arm

    @property
    def rgt_arm(self):
        # read-only property
        return self._rgt_arm

    def move_jnts(self, jnt_values):
        """
        move all joints of the ur3 dual-arm robot_s
        NOTE that the two arms are moved sequentially
        :param jnt_values: 1x12 array, depending on the value of component_name
        author: weiwei
        date: 20170411, 20240909
        """
        self._lft_arm.move_jnts(jnt_values[0:6], wait=False)
        self._rgt_arm.move_jnts(jnt_values[6:12], wait=True)

    def move_jspace_path(self, path, ctrl_freq=.008, max_vels=None, max_accs=None):
        """
        :param component_name
        :param path: a list of 1x12 arrays
        :param ctrl_freq:
        :param max_vels
        :param max_accs
        :return:
        author: weiwei
        date: 20210404, 20240909
        """
        # upload a urscript to connect to the pc server started by this class
        self._rgt_arm.arm.send_program(self._slave_modern_driver_urscript)
        self._lft_arm.arm.send_program(self._master_modern_driver_urscript)
        # accept arm socket
        pc_server_socket, pc_server_socket_addr = self._lft_arm.pc_server_socket.accept()
        print("PC server connected by ", pc_server_socket_addr)
        # send trajectory
        _, interp_confs, _, _ = trajp.generate_time_optimal_trajectory(path,
                                                                       max_vels=max_vels,
                                                                       max_accs=max_accs,
                                                                       ctrl_freq=ctrl_freq)
        keepalive = 1
        buf = bytes()
        for id, conf in enumerate(interp_confs):
            if id == len(interp_confs) - 1:
                keepalive = 0
            jointsradint = [int(jnt_value * self._lft_arm.jnts_scaler) for jnt_value in conf]
            buf += struct.pack('!iiiiiiiiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                               jointsradint[3], jointsradint[4], jointsradint[5], jointsradint[6],
                               jointsradint[7], jointsradint[8], jointsradint[9], jointsradint[10],
                               jointsradint[11], keepalive)
        pc_server_socket.send(buf)
        pc_server_socket.close()

    def get_jnt_values(self, component_name):
        """
        get the joint angles of both arms
        :return: 1x12 array
        author: ochi, revised by weiwei
        date: 20180410, 20210404
        """
        if component_name == "all":
            return np.array(self._lft_arm.get_jnt_values() + self._rgt_arm.get_jnt_values())
        elif component_name in ["lft_arm", "lft_hnd"]:
            return self._lft_arm.get_jnt_values()
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            return self._rgt_arm.get_jnt_values()
        else:
            raise ValueError("Component_name must be in ['all', 'lft_arm', 'rgt_arm']!")


if __name__ == '__main__':
    u3r85dx = UR3DualX(lft_robot_ip='10.2.0.50', rgt_robot_ip='10.2.0.51', pc_ip='10.2.0.100')
    u3r85dx.rgt_arm.close_gripper()
    u3r85dx.lft_arm.open_gripper()
