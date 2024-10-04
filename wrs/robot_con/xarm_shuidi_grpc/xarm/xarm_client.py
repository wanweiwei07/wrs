import grpc
import math
import numpy as np
import wrs.robot_con.xarm_shuidi_grpc.xarm.xarm_pb2 as xarm_msg
import wrs.robot_con.xarm_shuidi_grpc.xarm.xarm_pb2_grpc as xarm_rpc
import wrs.motion.trajectory.piecewisepoly_scl as pwply


class XArm7(object):

    def __init__(self, host="localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = xarm_rpc.XArmStub(channel)

    def get_jnt_vlaues(self):
        jntvalues_msg = self.stub.get_jnt_values(xarm_msg.Empty())
        jnt_values = np.frombuffer(jntvalues_msg.data, dtype=np.float64)
        return jnt_values

    def move_jspace_path(self, path, time_interval, toggle_debug=False):
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
        interpolated_path, interpolated_spd, interpolated_acc = tpply.interpolate_by_time_interval(path=path,
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
        path_msg = xarm_msg.Path(length=len(interpolated_path),
                                 njnts=len(interpolated_path[0]),
                                 data=np.array(interpolated_path).tobytes())
        return_value = self.stub.move_jspace_path(path_msg)
        if return_value == xarm_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The rbt_s has finished the given motion.")

    def get_jawwidth(self):
        gripper_msg = self.stub.get_gripper_status(xarm_msg.Empty())
        return (gripper_msg.position+10)/860

    def jaw_to(self, jawwidth, speed=None):
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
        gripper_msg = xarm_msg.GripperStatus(speed=speed,
                                             position=position)
        return_value = self.stub.jaw_to(gripper_msg)
        if return_value == xarm_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The grippers has finished the given action.")


if __name__ == "__main__":
    rbt_x = XArm7(host="192.168.50.77:18300")
    # path = [[0, 0, 0, 0, 0, 0, 0]]
    # rbt_x.move_jspace_path(path)
    # nxt.playPattern([anglesrad], [5.0])
    # nxt.goOffPose()
    init_jnt_angles = rbt_x.get_jnt_vlaues()
    print(init_jnt_angles)
    init_jawwidth = rbt_x.get_jawwidth()
    print(init_jawwidth)
    rbt_x.jaw_to(0)
