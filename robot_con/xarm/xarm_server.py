import grpc
import time
import numpy as np
import drivers.xarm.wrapper.xarm_api as xai
from concurrent import futures
import xarm_pb2 as xarm_msg
import xarm_pb2_grpc as xarm_rpc
import numpy as np


class XArmServer(xarm_rpc.XArmServicer):

    def initialize(self, xai_x):
        """
        :param xai_x: an instancde of xai.XArmAPI
        :return:
        """
        self._xai_x = xai_x

    def move_jspace_path(self, request, context):
        nrow = request.length
        ncol = request.njnts
        flat_path_data = np.frombuffer(request.data, dtype=np.float32)
        path = flat_path_data.reshape((nrow, ncol))
        print(path)
        # self._xai_x.set_servo_angle_j(path, is_radian=True)

def serve(robot_ip = "192.168.1.185", host = "10.2.0.170:18300"):
    xai_x = xai.XArmAPI(port=robot_ip)
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    xai_server = XArmServer(xai_x)
    xarm_rpc.add_PhoxiServicer_to_server(xai_server, server)
    server.add_insecure_port(host)
    server.start()
    print("The XArm server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve(robot_ip = "192.168.1.185", host = "10.2.0.170:18300")