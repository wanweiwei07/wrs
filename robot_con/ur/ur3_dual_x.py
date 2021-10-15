import time
import robot_con.ur.ur3_rtq85_x as u3r85x
import robot_con.ur.program_builder as pb
import struct
import os
import numpy as np


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
        self._lft_arm_hnd = u3r85x.UR3Rtq85X(robot_ip=lft_robot_ip, pc_ip=pc_ip)
        self._rgt_arm_hnd = u3r85x.UR3Rtq85X(robot_ip=rgt_robot_ip, pc_ip=pc_ip)
        self._pb = pb.ProgramBuilder()
        self._script_dir = os.path.dirname(__file__)
        self._pb.load_prog(os.path.join(self._script_dir, "urscripts_cbseries/moderndriver_cbseries_master.script"))
        self._master_modern_driver_urscript = self._pb.get_program_to_run()
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_pc_ip",
                                                                                          self._lft_arm_hnd.pc_server_socket_info[
                                                                                              0])
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_pc_port",
                                                                                          str(
                                                                                              self._lft_arm_hnd.pc_server_socket_info[
                                                                                                  1]))
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_slave_ip",
                                                                                          rgt_robot_ip)
        self._master_modern_driver_urscript = self._master_modern_driver_urscript.replace("parameter_jnts_scaler",
                                                                                          str(self._lft_arm_hnd.jnts_scaler))
        self._pb.load_prog(os.path.join(self._script_dir, "urscripts_cbseries/moderndriver_cbseries_slave.script"))
        self._slave_modern_driver_urscript = self._pb.get_program_to_run()
        self._slave_modern_driver_urscript = self._slave_modern_driver_urscript.replace("parameter_master_ip",
                                                                                        lft_robot_ip)
        self._slave_modern_driver_urscript = self._slave_modern_driver_urscript.replace("parameter_jnts_scaler",
                                                                                        str(self._lft_arm_hnd.jnts_scaler))
        print(self._slave_modern_driver_urscript)

    @property
    def lft_arm_hnd(self):
        # read-only property
        return self._lft_arm_hnd

    @property
    def rgt_arm_hnd(self):
        # read-only property
        return self._rgt_arm_hnd

    def move_jnts(self, component_name, jnt_values):
        """
        move all joints of the ur5 dual-arm robot_s
        NOTE that the two arms are moved sequentially
        :param component_name
        :param jnt_values: 1x6 or 1x12 array, depending on the value of component_name
        :return: bool

        author: weiwei
        date: 20170411
        """
        if component_name == "all": # TODO Examine length, synchronization
            self._lft_arm_hnd.move_jnts(jnt_values[0:6], wait=False)
            self._rgt_arm_hnd.move_jnts(jnt_values[6:12], wait=True)
        elif component_name in ["lft_arm", "lft_hnd"]:
            self._lft_arm_hnd.move_jnts(jnt_values, wait=False)
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            self._rgt_arm_hnd.move_jnts(jnt_values, wait=False)
        else:
            raise ValueError("Component_name must be in ['all', 'lft_arm', 'rgt_arm']!")

    def move_jspace_path(self,
                         component_name,
                         path,
                         control_frequency=.008,
                         interval_time=1.0,
                         interpolation_method=None):
        """
        :param component_name
        :param path: a list of 1x12 arrays or 1x6 arrays, depending on component_name
        :param control_frequency: the program will sample time_intervals/control_frequency confs, see motion.trajectory
        :param interval_time: equals to expandis/speed, speed = degree/second
                              by default, the value is 1.0 and the speed is expandis/second
        :param interpolation_method
        :return:
        author: weiwei
        date: 20210404
        """
        if component_name == "all":
            self._lft_arm_hnd.trajt.set_interpolation_method(interpolation_method)
            interpolated_confs, interpolated_spds = self._lft_arm_hnd.trajt.piecewise_interpolation(path,
                                                                                                    control_frequency,
                                                                                                    interval_time)
            # upload a urscript to connect to the pc server started by this class
            self._rgt_arm_hnd.arm.send_program(self._slave_modern_driver_urscript)
            self._lft_arm_hnd.arm.send_program(self._master_modern_driver_urscript)
            # accept arm socket
            pc_server_socket, pc_server_socket_addr = self._lft_arm_hnd.pc_server_socket.accept()
            print("PC server connected by ", pc_server_socket_addr)
            # send trajectory
            keepalive = 1
            buf = bytes()
            for id, conf in enumerate(interpolated_confs):
                if id == len(interpolated_confs) - 1:
                    keepalive = 0
                jointsradint = [int(jnt_value * self._lft_arm_hnd.jnts_scaler) for jnt_value in conf]
                buf += struct.pack('!iiiiiiiiiiiii', jointsradint[0], jointsradint[1], jointsradint[2],
                                   jointsradint[3], jointsradint[4], jointsradint[5], jointsradint[6],
                                   jointsradint[7], jointsradint[8], jointsradint[9], jointsradint[10],
                                   jointsradint[11], keepalive)
            pc_server_socket.send(buf)
            pc_server_socket.close()
        elif component_name in ["lft_arm", "lft_hnd"]:
            self._lft_arm_hnd.move_jntspace_path(path=path,
                                                 control_frequency=control_frequency,
                                                 interval_time=interval_time,
                                                 interpolation_method=interpolation_method)
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            self._rgt_arm_hnd.move_jntspace_path(path=path,
                                                 control_frequency=control_frequency,
                                                 interval_time=interval_time,
                                                 interpolation_method=interpolation_method)
        else:
            raise ValueError("Component_name must be in ['all', 'lft_arm', 'rgt_arm']!")

    def get_jnt_values(self, component_name):
        """
        get the joint angles of both arms
        :return: 1x12 array
        author: ochi, revised by weiwei
        date: 20180410, 20210404
        """
        if component_name == "all":
            return np.array(self._lft_arm_hnd.get_jnt_values() + self._rgt_arm_hnd.get_jnt_values())
        elif component_name in ["lft_arm", "lft_hnd"]:
            return self._lft_arm_hnd.get_jnt_values()
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            return self._rgt_arm_hnd.get_jnt_values()
        else:
            raise ValueError("Component_name must be in ['all', 'lft_arm', 'rgt_arm']!")


if __name__ == '__main__':
    u3r85dx = UR3DualX(lft_robot_ip='10.2.0.50', rgt_robot_ip='10.2.0.51', pc_ip='10.2.0.100')
    u3r85dx.rgt_arm_hnd.open_gripper()
    time.sleep(2)
    u3r85dx.lft_arm_hnd.close_gripper()
