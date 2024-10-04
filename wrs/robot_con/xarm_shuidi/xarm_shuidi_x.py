import math
import time
import numpy as np
import shuidi_x as agv
import wrs.motion.trajectory.piecewisepoly_toppra as pwp


class XArmShuidiX(object):

    def __init__(self, ip="10.2.0.203"):
        """
        :param _arm_x: an instancde of arm.XArmAPI
        :return:
        """
        super().__init__()
        self._arm_x = arm.XArmAPI(port=ip)
        if self._arm_x.has_err_warn:
            if self._arm_x.get_err_warn_code()[1][0] == 1:
                print("The Emergency Button is pushed in to stop!")
                input("Release the emergency button and press any key to continue. Press Enter to continue...")
        self._arm_x.clean_error()
        self._arm_x.clean_error()
        self._arm_x.motion_enable()
        self._arm_x.set_mode(1)  # servo motion mode
        self._arm_x.set_state(state=0)
        self._arm_x.reset(wait=True)
        self._arm_x.clean_gripper_error()
        self._arm_x.set_gripper_enable(1)
        self._arm_x.set_gripper_mode(0)
        self.__speed = 5000
        self._arm_x.set_gripper_speed(self.__speed)  # 1000-5000
        self._arm_x.set_gripper_position(850)  # 1000-5000
        self._agv_x = agv.ShuidiX(ip=ip)
        print("The Shuidi server is started!")

    @property
    def arm(self):
        return self._arm_x

    @property
    def agv(self):
        return self._agv_x

    def arm_get_jnt_values(self):
        code, jnt_values = self._arm_x.get_servo_angle(is_radian=True)
        if code != 0:
            raise Exception(f"The returned code of get_servo_angle is wrong! Code: {code}")
        return np.asarray(jnt_values)

    def arm_move_jspace_path(self,
                             path,
                             max_jntvel=None,
                             max_jntacc=None,
                             start_frame_id=1,
                             toggle_debug=False):
        """
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :param max_jntvel:
        :param max_jntacc:
        :param start_frame_id:
        :return:
        """
        if not path or path is None:
            raise ValueError("The given is incorrect!")
        control_frequency = .005
        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                            control_frequency=control_frequency,
                                                            max_vels=max_jntvel,
                                                            max_accs=max_jntacc,
                                                            toggle_debug=toggle_debug)
        interpolated_path = interpolated_path[start_frame_id:]
        for jnt_values in interpolated_path:
            self._arm_x.set_servo_angle_j(jnt_values, is_radian=True)
        return

    def arm_jaw_to(self, jawwidth, speed=None):
        position = math.floor(860 * jawwidth / 100) - 10
        if speed is None:
            speed = 5000
        else:
            speed = math.floor(5000 * speed / 100)
        self.__speed = speed
        self._arm_x.set_gripper_speed(self.__speed)
        self._arm_x.set_gripper_position(position, wait=True)
        return

    def arm_get_jaw_width(self):
        code, position = self._arm_x.get_gripper_position()
        if code != 0:
            raise Exception(f"The returned code of get_gripper_position is wrong! Code: {code}")
        return (position + 10) / 860 * .085

    def agv_move(self, linear_speed=.0, angular_speed=.0, time_interval=.5):
        while time_interval > 0:
            # try:
            self._agv_x.joy_control(linear_velocity=linear_speed,
                                    angular_velocity=angular_speed)
            time_interval = time_interval - .5
            time.sleep(.3)
        return

if __name__ == "__main__":
    import keyboard
    from wrs import basis as rm, drivers as arm, robot_sim as rbt
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 1, 1.5], lookat_pos=[0, 0, 0.7])
    rbt_s = rbt.XArmShuidi()
    rbt_x = XArmShuidiX(ip="10.2.0.203")
    jnt_values = rbt_x.arm_get_jnt_values()
    print(jnt_values)
    jawwidth = rbt_x.arm_get_jaw_width()
    rbt_s.fk(jnt_values=jnt_values)
    rbt_s.jaw_to(jawwidth=jawwidth)
    rbt_s.gen_meshmodel().attach_to(base)
    # base.run()
    # rbt_x.agv_move(agv_linear_speed=-.1, agv_angular_speed=.1, time_intervals=5)
    agv_linear_speed = .2
    agv_angular_speed = .5
    arm_linear_speed = .02
    arm_angular_speed = .05
    while True:
        pressed_keys = {"w": keyboard.is_pressed('w'),
                        "a": keyboard.is_pressed('a'),
                        "s": keyboard.is_pressed('s'),
                        "d": keyboard.is_pressed('d'),
                        "r": keyboard.is_pressed('r'),  # x+ global
                        "t": keyboard.is_pressed('t'),  # x- global
                        "f": keyboard.is_pressed('f'),  # y+ global
                        "g": keyboard.is_pressed('g'),  # y- global
                        "v": keyboard.is_pressed('v'),  # z+ global
                        "b": keyboard.is_pressed('b'),  # z- global
                        "y": keyboard.is_pressed('y'),  # r+ global
                        "u": keyboard.is_pressed('u'),  # r- global
                        "h": keyboard.is_pressed('h'),  # p+ global
                        "j": keyboard.is_pressed('j'),  # p- global
                        "n": keyboard.is_pressed('n'),  # yaw+ global
                        "m": keyboard.is_pressed('m')}  # yaw- global
        # "R": keyboard.is_pressed('R'),  # x+ local
        # "T": keyboard.is_pressed('T'),  # x- local
        # "F": keyboard.is_pressed('F'),  # y+ local
        # "G": keyboard.is_pressed('G'),  # y- local
        # "V": keyboard.is_pressed('V'),  # z+ local
        # "B": keyboard.is_pressed('B'),  # z- local
        # "Y": keyboard.is_pressed('Y'),  # r+ local
        # "U": keyboard.is_pressed('U'),  # r- local
        # "H": keyboard.is_pressed('H'),  # p+ local
        # "J": keyboard.is_pressed('J'),  # p- local
        # "N": keyboard.is_pressed('N'),  # yaw+ local
        # "M": keyboard.is_pressed('M')}  # yaw- local
        values_list = list(pressed_keys.values())
        if pressed_keys["w"] and pressed_keys["a"]:
            rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
        elif pressed_keys["w"] and pressed_keys["d"]:
            rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
        elif pressed_keys["s"] and pressed_keys["a"]:
            rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=-agv_angular_speed, time_interval=.5)
        elif pressed_keys["s"] and pressed_keys["d"]:
            rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=agv_angular_speed, time_interval=.5)
        elif pressed_keys["w"] and sum(values_list) == 1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=agv_linear_speed, angular_speed=.0, time_interval=.5)
        elif pressed_keys["s"] and sum(values_list) == 1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=-agv_linear_speed, angular_speed=.0, time_interval=.5)
        elif pressed_keys["a"] and sum(values_list) == 1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=.0, angular_speed=agv_angular_speed, time_interval=.5)
        elif pressed_keys["d"] and sum(values_list) == 1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=.0, angular_speed=-agv_angular_speed, time_interval=.5)
        elif any(pressed_keys[item] for item in ['r', 't', 'f', 'g', 'v', 'b', 'y', 'u', 'h', 'j', 'n', 'm']) and \
                sum(values_list) == 1:  # global
            tic = time.time()
            current_arm_tcp_pos, current_arm_tcp_rotmat = rbt_s.get_gl_tcp()
            rel_pos = np.zeros(3)
            rel_rotmat = np.eye(3)
            if pressed_keys['r']:
                rel_pos = np.array([arm_linear_speed * .5, 0, 0])
            elif pressed_keys['t']:
                rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
            elif pressed_keys['f']:
                rel_pos = np.array([0, arm_linear_speed * .5, 0])
            elif pressed_keys['g']:
                rel_pos = np.array([0, -arm_linear_speed * .5, 0])
            elif pressed_keys['v']:
                rel_pos = np.array([0, 0, arm_linear_speed * .5])
            elif pressed_keys['b']:
                rel_pos = np.array([0, 0, -arm_linear_speed * .5])
            elif pressed_keys['y']:
                rel_rotmat = rm.rotmat_from_euler(arm_angular_speed * .5, 0, 0)
            elif pressed_keys['u']:
                rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed * .5, 0, 0)
            elif pressed_keys['h']:
                rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed * .5, 0)
            elif pressed_keys['j']:
                rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
            elif pressed_keys['n']:
                rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed * .5)
            elif pressed_keys['m']:
                rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed * .5)
            new_arm_tcp_pos = current_arm_tcp_pos + rel_pos
            new_arm_tcp_rotmat = rel_rotmat.dot(current_arm_tcp_rotmat)
            last_jnt_values = rbt_s.get_jnt_values()
            new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos,
                                      tgt_rotmat=new_arm_tcp_rotmat,
                                      seed_jnt_values=last_jnt_values)
            if new_jnt_values is not None:
                print(new_jnt_values)
                print(last_jnt_values)
                max_change = np.max(new_jnt_values-last_jnt_values)
                print(max_change)
                # rbt_s.fk(jnt_values=new_jnt_values)
                # rbt_s.jaw_to(ee_values=ee_values)
                # rbt_s.gen_meshmodel().attach_to(base)
                # base.run()
            else:
                continue
            rbt_s.fk(jnt_values=new_jnt_values)
            toc = time.time()
            start_frame_id = math.ceil((toc - tic) / .01)
            rbt_x.arm_move_jspace_path([last_jnt_values, new_jnt_values],
                                       start_frame_id=start_frame_id)