import grpc
import time
from wrs import drivers as xai
from concurrent import futures
import wrs.robot_con.xarm_shuidi_grpc.xarm.xarm_pb2 as xarm_msg
import wrs.robot_con.xarm_shuidi_grpc.xarm.xarm_pb2_grpc as xarm_rpc
import numpy as np


class XArmServer(xarm_rpc.XArmServicer):

    def __init__(self, arm_ip):
        """
        :param _arm_x: an instancde of arm.XArmAPI
        :return:
        """
        super().__init__()
        self._xai_x = xai.XArmAPI(port=arm_ip)
        if self._xai_x.has_err_warn:
            if self._xai_x.get_err_warn_code()[1][0] == 1:
                print("The Emergency Button is pushed in to stop!")
                input("Release the emergency button and press any key to continue. Press Enter to continue...")
        self._xai_x.clean_error()
        self._xai_x.clean_error()
        self._xai_x.motion_enable()
        self._xai_x.set_mode(1)  # servo motion mode
        self._xai_x.set_state(state=0)
        self._xai_x.reset(wait=True)
        self._xai_x.clean_gripper_error()
        self._xai_x.set_gripper_enable(1)
        self._xai_x.set_gripper_mode(0)
        self.__speed = 5000
        self._xai_x.set_gripper_speed(self.__speed) # 1000-5000
        self._xai_x.set_gripper_position(850) # 1000-5000

    def get_jnt_values(self, request, context):
        code, jnt_values = self._xai_x.get_servo_angle(is_radian=True)
        if code != 0:
            raise Exception(f"The returned code of get_servo_angle is wrong! Code: {code}")
        return xarm_msg.JntValues(data=np.array(jnt_values).tobytes())

    def move_jspace_path(self, request, context):
        nrow = request.length
        ncol = request.njnts
        flat_path_data = np.frombuffer(request.data, dtype=np.float64)
        path = flat_path_data.reshape((nrow, ncol))
        for jnt_values in path.tolist():
            self._xai_x.set_servo_angle_j(jnt_values, is_radian=True)
            time.sleep(.01)
        return xarm_msg.Status(value=xarm_msg.Status.DONE)

    def jaw_to(self, request, context):
        self.__speed = request.speed
        self._xai_x.set_gripper_speed(self.__speed)
        self._xai_x.set_gripper_position(request.position, wait=True)
        return xarm_msg.Status(value=xarm_msg.Status.DONE)

    def get_gripper_status(self, request, context):
        speed = self.__speed
        code, position = self._xai_x.get_gripper_position()
        if code != 0:
            raise Exception(f"The returned code of get_gripper_position is wrong! Code: {code}")
        return xarm_msg.GripperStatus(speed=speed,
                                      position=position)

def serve(arm_ip = "192.168.50.99", host = "localhost:18300"):
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    xai_server = XArmServer(arm_ip)
    xarm_rpc.add_XArmServicer_to_server(xai_server, server)
    server.add_insecure_port(host)
    server.start()
    print("The XArm server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve(arm_ip = "192.168.1.185", host = "192.168.50.99:18300")