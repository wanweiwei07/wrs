import struct
import numpy as np
import wrs.robot_con.ur.ur3e_rtqhe_x as u3erhex

class Ur3EDualUrx(object):
    """
    urx 50, right arm 51, left arm 52
    author: weiwei
    date: 20180131
    """

    def __init__(self, lft_robot_ip='10.0.2.2', rgt_robot_ip='10.0.2.3', pc_ip='10.2.0.100'):
        """

        :param robotsim: for global transformation, especially in attachfirm

        author: weiwei
        date: 20191014 osaka
        """
        self._lft_arm_hnd = u3erhex.UR3ERtqHEX(robot_ip=lft_robot_ip, pc_ip=pc_ip)
        self._rgt_arm_hnd = u3erhex.UR3ERtqHEX(robot_ip=rgt_robot_ip, pc_ip=pc_ip)

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
        if component_name == "all":  # TODO Examine axis_length, synchronization
            self._lft_arm_hnd.move_jnts(jnt_values[0:6], wait=False)
            self._rgt_arm_hnd.move_jnts(jnt_values[6:12], wait=True)
        elif component_name in ["lft_arm", "lft_hnd"]:
            self._lft_arm_hnd.move_jnts(jnt_values, wait=False)
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            self._rgt_arm_hnd.move_jnts(jnt_values, wait=False)
        else:
            raise ValueError("Component_name must be in ['all', 'lft_arm', 'rgt_arm']!")

    def move_jntspace_path(self,
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
            if interpolation_method:
                self._lft_arm_hnd.trajt.change_method(interpolation_method)
            interpolated_confs, _, _, _ = self._lft_arm_hnd.trajt.interpolate_by_time_interval(path,
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
            self._lft_arm_hnd.move_jspace_path(path=path,
                                               control_frequency=control_frequency,
                                               interval_time=interval_time,
                                               interpolation_method=interpolation_method)
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            self._rgt_arm_hnd.move_jspace_path(path=path,
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
    from wrs import robot_sim as u3ed
    import pandaplotutils.pandactrl as pc

    base = pc.World(camp=[3000, 0, 3000], lookatp=[0, 0, 700])

    ur3edualrobot = u3ed.Ur3EDualRobot()
    ur3edualrobot.goinitpose()
    ur3eu = Ur3EDualUrx(ur3edualrobot)
    #
    # hndfa = rtqhe.RobotiqHEFactory()
    # rgthnd = hndfa.genHand()
    # lfthnd = hndfa.genHand()
    #
    # robot_s = robot_s.Ur3EDualRobot(rgthnd, lfthnd)
    # robot_s.goinitpose()
    # ur3eu.attachfirm(robot_s, upthreshold=10, arm_name='lft')
    ur3eu.opengripper(armname="lft", forcepercentage=0, distance=23)
    ur3eu.opengripper(armname="lft", forcepercentage=0, distance=80)
    # ur3eu.closegripper(arm_name="lft")
    # initpose = ur3dualrobot.initjnts
    # initrgt = initpose[3:9]
    # initlft = initpose[9:15]
    # ur3u.movejntssgl(initrgt, arm_name='rgt')
    # ur3u.movejntssgl(initlft, arm_name='lft')

    # goalrgt = copy.deepcopy(initrgt)
    # goalrgt[0] = goalrgt[0]-10.0
    # goalrgt1 = copy.deepcopy(initrgt)zr
    # goalrgt1[0] = goalrgt1[0]-5.0
    # goallft = copy.deepcopy(initlft)
    # goallft[0] = goallft[0]+10.0

    # ur3u.movejntssgl_cont([initrgt, goalrgt, goalrgt1], arm_name='rgt')
    #
    # postcp_robot, rottcp_robot =  ur3dualrobot.gettcp_robot(arm_name='rgt')
    # print math3d.Transform(rottcp_robot, postcp_robot).get_pose_vector()
    # print "getl ", ur3u.rgtarm.getl()
    #
    # postcp_robot, rottcp_robot =  ur3dualrobot.gettcp_robot(arm_name='lft')
    # print math3d.Transform(rottcp_robot, postcp_robot).get_pose_vector()
    # print "getl ", ur3u.lftarm.getl()
    #
    # # tcpsimrobot =  ur5dualrobot.lftarm[-1]['linkpos']
    # # print tcprobot
    # # print tcpsimrobot
    # u3dmgen = u3dm.Ur3DualMesh(rgthand, lfthand)
    # ur3dualmnp = u3dmgen.genmnp(ur3dualrobot, togglejntscoord=True)
    # ur3dualmnp.reparentTo(base.render)
    # # arm_name = 'rgt'
    # # arm_name = 'lft'
    # # ur3u.movejntsall(ur3dualrobot.initjnts)
    # # ur3u.movejntsin360()
    # # print ur3u.getjnts('rgt')
    # # print ur3u.getjnts('lft')

    base.run()
