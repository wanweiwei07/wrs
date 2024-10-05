import grpc
import time
from concurrent import futures
import wrs.drivers.xarm.wrapper.xarm_api as arm
import wrs.robot_con.xarm_shuidi_grpc.shuidi.shuidi_robot as agv
import wrs.robot_con.xarm_shuidi_grpc.xarm_shuidi_pb2 as aa_msg  # aa = arm_agv
import wrs.robot_con.xarm_shuidi_grpc.xarm_shuidi_pb2_grpc as aa_rpc
import numpy as np


class XArmShuidiServer(aa_rpc.XArmShuidiServicer):

    def __init__(self, arm_ip, agv_ip="192.168.10.10"):
        """
        :param _arm_x: an instancde of arm.XArmAPI
        :return:
        """
        super().__init__()
        self._arm_x = arm.XArmAPI(port=arm_ip)
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
        self._agv_x = agv.ShuidiRobot(ip=agv_ip)
        print("The Shuidi server is started!")

    def arm_get_jnt_values(self, request, context):
        code, jnt_values = self._arm_x.get_servo_angle(is_radian=True)
        if code != 0:
            raise Exception(f"The returned code of get_servo_angle is wrong! Code: {code}")
        return aa_msg.JntValues(data=np.array(jnt_values).tobytes())

    def arm_move_jspace_path(self, request, context):
        nrow = request.length
        ncol = request.njnts
        flat_path_data = np.frombuffer(request.data, dtype=np.float64)
        path = flat_path_data.reshape((nrow, ncol))
        for jnt_values in path.tolist():
            self._arm_x.set_servo_angle_j(jnt_values, is_radian=True)
            time.sleep(.01)
        return aa_msg.Status(value=aa_msg.Status.DONE)

    def arm_jaw_to(self, request, context):
        self.__speed = request.speed
        self._arm_x.set_gripper_speed(self.__speed)
        self._arm_x.set_gripper_position(request.position, wait=True)
        return aa_msg.Status(value=aa_msg.Status.DONE)

    def arm_get_gripper_status(self, request, context):
        speed = self.__speed
        code, position = self._arm_x.get_gripper_position()
        if code != 0:
            raise Exception(f"The returned code of get_gripper_position is wrong! Code: {code}")
        return aa_msg.GripperStatus(speed=speed,
                                    position=position)

    def agv_move(self, request, context):
        linear_speed = request.linear_velocity
        angular_speed = request.angular_velocity
        self._agv_x.joy_control(linear_velocity=linear_speed,
                                angular_velocity=angular_speed)
        return aa_msg.Status(value=aa_msg.Status.DONE)


def serve(arm_ip="192.168.50.99", agv_ip="192.168.10.10", host="localhost:18300"):
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    aa_server = XArmShuidiServer(arm_ip=arm_ip, agv_ip=agv_ip)
    aa_rpc.add_XArmShuidiServicer_to_server(aa_server, server)
    server.add_insecure_port(host)
    server.start()
    print("The XArm server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve(arm_ip="192.168.1.185", host="192.168.50.99:18300")
