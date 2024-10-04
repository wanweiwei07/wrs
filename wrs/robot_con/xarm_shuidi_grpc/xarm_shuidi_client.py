import grpc
import math
import time
import numpy as np
import wrs.robot_con.xarm_shuidi_grpc.xarm_shuidi_pb2 as aa_msg
import wrs.robot_con.xarm_shuidi_grpc.xarm_shuidi_pb2_grpc as aa_rpc
import wrs.motion.trajectory.piecewisepoly_toppra as pwp


class XArmShuidiClient(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = aa_rpc.XArmShuidiStub(channel)

    def get_jnt_values(self, component_name="arm"):
        if component_name == "arm":
            return self.arm_get_jnt_values()

    def move_jnts(self, component_name, jnt_values, method='linear', max_jntspeed=math.pi):
        """
        TODO: use xarm function to get faster
        author: weiwei
        date: 20210729
        """
        if component_name == "arm":
            current_jnt_values = self.arm_get_jnt_values()
            print(current_jnt_values, jnt_values)
            if np.allclose(jnt_values, current_jnt_values, atol=1e-5):
                print("The robot's configuration is the same as the given one!")
                return
            self.arm_move_jspace_path(path=[self.arm_get_jnt_values(), jnt_values], method=method,
                                      max_jntspeed=max_jntspeed)

    def arm_get_jnt_values(self):
        jntvalues_msg = self.stub.arm_get_jnt_values(aa_msg.Empty())
        jnt_values = np.frombuffer(jntvalues_msg.data, dtype=np.float64)
        return jnt_values

    def arm_move_jspace_path(self,
                             path,
                             max_jntvel=None,
                             max_jntacc=None,
                             start_frame_id=1):
        """
        TODO: make speed even
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :return:
        author: weiwei
        date: 20190417
        """
        if not path or path is None:
            raise ValueError("The given is incorrect!")
        control_frequency = .005
        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated_path = tpply.interpolate_by_max_spdacc(path=path, control_frequency=control_frequency,
                                                            max_jntvel=max_jntvel, max_jntacc=max_jntacc)
        interpolated_path = interpolated_path[start_frame_id:]
        path_msg = aa_msg.Path(length=len(interpolated_path),
                               njnts=len(interpolated_path[0]),
                               data=np.array(interpolated_path).tobytes())
        return_value = self.stub.arm_move_jspace_path(path_msg)
        if return_value == aa_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The rbt_s has finished the given motion.")

    def arm_get_jawwidth(self):
        gripper_msg = self.stub.arm_get_gripper_status(aa_msg.Empty())
        return (gripper_msg.position + 10) / 860 * .085

    def arm_jaw_to(self, jawwidth, speed=None):
        """
        both values are in percentage
        :param jawwidth: 0~100
        :param speed: 0~100
        :return:
        """
        if speed is None:
            speed = 5000
        else:
            speed = math.floor(5000 * speed / 100)
        position = math.floor(860 * jawwidth / 100) - 10
        gripper_msg = aa_msg.GripperStatus(speed=speed,
                                           position=position)
        return_value = self.stub.arm_jaw_to(gripper_msg)
        if return_value == aa_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The grippers has finished the given action.")

    def agv_move(self, linear_speed=.0, angular_speed=.0, time_interval=.5):
        while time_interval > 0:
            speed_msg = aa_msg.Speed(linear_velocity=linear_speed,
                                     angular_velocity=angular_speed)
            # try:
            return_value = self.stub.agv_move(speed_msg)
            if return_value == aa_msg.Status.ERROR:
                print("Something went wrong with the server!!")
                continue
            time_interval = time_interval - .5
            time.sleep(.3)
            # except Exception:
            #     pass


if __name__ == "__main__":
    import keyboard
    from wrs import basis as rm, robot_sim as rbt
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[3, 1, 1.5], lookat_pos=[0, 0, 0.7])
    rbt_s = rbt.XArmShuidi()
    rbt_x = XArmShuidiClient(host="10.2.0.201:18300")
    jnt_values = rbt_x.arm_get_jnt_vlaues()
    jawwidth = rbt_x.arm_get_jawwidth()
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
            new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos, tgt_rotmat=new_arm_tcp_rotmat)
            rbt_s.fk(jnt_values=new_jnt_values)
            toc = time.time()
            start_frame_id = math.ceil((toc - tic) / .01)
            rbt_x.arm_move_jspace_path([last_jnt_values, new_jnt_values], time_interval=.1,
                                       start_frame_id=start_frame_id)
        # elif any(pressed_keys[item] for item in ['R', 'T', 'F', 'G', 'V', 'B', 'Y', 'U', 'H', 'J', 'N', 'M']) and\
        #         sum(values_list) == 1: # local
        #     tic = time.time()
        #     rel_pos = np.zeros(3)
        #     rel_rotmat = np.eye(3)
        #     if pressed_keys['r']:
        #         rel_pos = np.array([arm_linear_speed * .5, 0, 0])
        #     elif pressed_keys['t']:
        #         rel_pos = np.array([-arm_linear_speed * .5, 0, 0])
        #     elif pressed_keys['f']:
        #         rel_pos = np.array([0, arm_linear_speed * .5, 0])
        #     elif pressed_keys['g']:
        #         rel_pos = np.array([0, -arm_linear_speed * .5, 0])
        #     elif pressed_keys['v']:
        #         rel_pos = np.array([0, 0, arm_linear_speed * .5])
        #     elif pressed_keys['b']:
        #         rel_pos = np.array([0, 0, -arm_linear_speed * .5])
        #     elif pressed_keys['y']:
        #         rel_rotmat = rm.rotmat_from_euler(arm_angular_speed*.5, 0, 0)
        #     elif pressed_keys['u']:
        #         rel_rotmat = rm.rotmat_from_euler(-arm_angular_speed*.5, 0, 0)
        #     elif pressed_keys['h']:
        #         rel_rotmat = rm.rotmat_from_euler(0, arm_angular_speed*.5, 0)
        #     elif pressed_keys['j']:
        #         rel_rotmat = rm.rotmat_from_euler(0, -arm_angular_speed * .5, 0)
        #     elif pressed_keys['n']:
        #         rel_rotmat = rm.rotmat_from_euler(0, 0, arm_angular_speed*.5)
        #     elif pressed_keys['m']:
        #         rel_rotmat = rm.rotmat_from_euler(0, 0, -arm_angular_speed*.5)
        #     new_arm_tcp_pos, new_arm_tcp_rotmat = rbt_s.cvt_loc_tcp_to_gl("arm",
        #                                                                   rel_obj_pos=rel_pos,
        #                                                                   rel_obj_rotmat=rel_rotmat)
        #     last_jnt_values = rbt_s.get_jnt_values()
        #     new_jnt_values = rbt_s.ik(tgt_pos=new_arm_tcp_pos, tgt_rotmat=new_arm_tcp_rotmat)
        #     rbt_s.fk(jnt_values=new_jnt_values)
        #     toc = time.time()
        #     start_frame_id = math.ceil((toc - tic) / .01)
        #     rbt_x.arm_move_jspace_path([last_jnt_values, new_jnt_values], time_intervals=.1, start_frame_id=start_frame_id)

# path = [[0, 0, 0, 0, 0, 0, 0]]wwwwwwwwwwww
# rbt_x.move_jspace_path(path)
# nxt.playPattern([anglesrad], [5.0])
# nxt.goOffPose()
# init_jnt_angles = rbt_x.get_jnt_vlaues()
# print(init_jnt_angles)
# init_jawwidth = rbt_x.get_jawwidth()
# print(init_jawwidth)
# rbt_x.jaw_to(0)
