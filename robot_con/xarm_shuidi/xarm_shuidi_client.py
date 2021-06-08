import keyword

import grpc
import math
import time
import numpy as np
import robot_con.xarm_shuidi.xarm_shuidi_pb2 as aa_msg
import robot_con.xarm_shuidi.xarm_shuidi_pb2_grpc as aa_rpc
import motion.trajectory.piecewisepoly as pwply


class XArmShuidiClient(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = aa_rpc.XArmShuidiStub(channel)

    def arm_get_jnt_vlaues(self):
        jntvalues_msg = self.stub.arm_get_jnt_values(aa_msg.Empty())
        jnt_values = np.frombuffer(jntvalues_msg.data, dtype=np.float64)
        return jnt_values

    def arm_move_jspace_path(self, path, time_interval, toggle_debug=False):
        """
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :return:
        author: weiwei
        date: 20190417
        """
        if not path or path is None:
            raise ValueError("The given is incorrect!")
        control_frequency = .005
        tpply = pwply.PiecewisePoly(method='linear')
        interpolated_path, interpolated_spd, interpolated_acc = tpply.interpolate(path=path,
                                                                                  control_frequency=control_frequency,
                                                                                  time_interval=time_interval)
        if toggle_debug:
            import matplotlib.pyplot as plt
            # plt.plot(interplated_path)
            samples = np.linspace(0,
                                  time_interval,
                                  math.floor(time_interval / control_frequency),
                                  endpoint=False) / time_interval
            nsample = len(samples)
            plt.subplot(311)
            for i in range(len(path)):
                plt.axvline(x=(nsample - 1) * i)
            plt.plot(interpolated_path)
            plt.subplot(312)
            for i in range(len(path)):
                plt.axvline(x=(nsample - 1) * i)
            plt.plot(interpolated_spd)
            plt.subplot(313)
            for i in range(len(path)):
                plt.axvline(x=(nsample - 1) * i)
            plt.plot(interpolated_acc)
            plt.show()
            import pickle
            pickle.dump([interpolated_path, interpolated_spd, interpolated_acc], open("interpolated_traj.pkl", "wb"))
        path_msg = aa_msg.Path(length=len(interpolated_path),
                               njnts=len(interpolated_path[0]),
                               data=np.array(interpolated_path).tobytes())
        return_value = self.stub.arm_move_jspace_path(path_msg)
        if return_value == aa_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot has finished the given motion.")

    def arm_get_jawwidth(self):
        gripper_msg = self.stub.arm_get_gripper_status(aa_msg.Empty())
        return (gripper_msg.position+10)/860

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
            print("The gripper has finished the given action.")

    def agv_move(self, linear_speed=0, angular_speed=0, time_interval=.5):
        while time_interval > 0:
            speed_msg = aa_msg.Speed(linear_velocity=linear_speed,
                                     angular_velocity=angular_speed)
            self.stub.agv_move(speed_msg)
            time_interval=time_interval-.5
            time.sleep(.3)

if __name__ == "__main__":
    import keyboard
    rbt_x = XArmShuidiClient(host="10.2.0.203:18300")
    # rbt_x.agv_move(linear_speed=-.1, angular_speed=.1, time_interval=5)
    while True:
        pressed_keys = {"w": keyboard.is_pressed('w'),
                        "a": keyboard.is_pressed('a'),
                        "s": keyboard.is_pressed('s'),
                        "d": keyboard.is_pressed('d')}
        values_list = list(pressed_keys.values())
        linear_speed = .2
        angular_speed = .5
        if pressed_keys["w"] and pressed_keys["a"]:
            rbt_x.agv_move(linear_speed=linear_speed, angular_speed=angular_speed, time_interval=.5)
        elif pressed_keys["w"] and pressed_keys["d"]:
            rbt_x.agv_move(linear_speed=linear_speed, angular_speed=-angular_speed, time_interval=.5)
        elif pressed_keys["s"] and pressed_keys["a"]:
            rbt_x.agv_move(linear_speed=-linear_speed, angular_speed=-angular_speed, time_interval=.5)
        elif pressed_keys["s"] and pressed_keys["d"]:
            rbt_x.agv_move(linear_speed=-linear_speed, angular_speed=angular_speed, time_interval=.5)
        elif pressed_keys["w"] and sum(values_list)==1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=linear_speed, angular_speed=0, time_interval=.5)
        elif pressed_keys["s"] and sum(values_list)==1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=-linear_speed, angular_speed=0, time_interval=.5)
        elif pressed_keys["a"] and sum(values_list)==1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=0, angular_speed=angular_speed, time_interval=.5)
        elif pressed_keys["d"] and sum(values_list)==1:  # if key 'q' is pressed
            rbt_x.agv_move(linear_speed=0, angular_speed=-angular_speed, time_interval=.5)
    # path = [[0, 0, 0, 0, 0, 0, 0]]wwwwwwwwwwww
    # rbt_x.move_jspace_path(path)
    # nxt.playPattern([anglesrad], [5.0])
    # nxt.goOffPose()
    # init_jnt_angles = rbt_x.get_jnt_vlaues()
    # print(init_jnt_angles)
    # init_jawwidth = rbt_x.get_jawwidth()
    # print(init_jawwidth)
    # rbt_x.jaw_to(0)
