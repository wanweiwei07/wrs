import grpc
import time
import numpy as np
from concurrent import futures
import visualization.panda.rpc.rviz_pb2 as rv_msg
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc
import visualization.panda.world as wd
import visualization.panda.rpc.render_info as rdi


def create_robot_render_info(robot_instance,
                             robot_jlc_name,
                             robot_meshmodel_parameters,
                             robot_path):
    robot_render_info = rdi.RobotInfo()
    robot_render_info.robot_instance = robot_instance
    robot_render_info.robot_jlc_name = robot_jlc_name
    robot_render_info.robot_meshmodel = robot_instance.gen_meshmodel(robot_meshmodel_parameters)
    robot_render_info.robot_meshmodel_parameters = robot_meshmodel_parameters
    robot_render_info.robot_path = robot_path
    robot_render_info.robot_path_counter = 0
    return robot_render_info


def create_obj_render_info(obj,
                           obj_parameters=None,
                           obj_path=None):
    obj_render_info = rdi.ObjInfo()
    obj_render_info.obj = obj
    if obj_parameters is None:
        obj_render_info.obj_parameters = obj.get_rgba()
    else:
        obj_render_info.obj_parameters = obj_parameters
    if obj_path is None:
        obj_render_info.obj_path = [[obj.get_pos(), obj.get_rotmat()]]
    else:
        obj_render_info.obj_path = obj_path
    obj_render_info.obj_path_counter = 0
    return obj_render_info


class RVizServer(rv_rpc.RVizServicer):

    def initialize(self):
        self.base = wd.World(campos=[1, 1, 1], lookatpos=[0, 0, 0])
        self.obj_dict = {}

    def run_code(self, request, context):
        """
        author: weiwei
        date: 20201229
        """
        try:
            code = request.code.decode('utf-8')
            for i, line in enumerate(code.splitlines()):
                print("{:< 4d}".format(i), ": ", line)
            exec(code, globals())
            return rv_msg.Status(value=rv_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return rv_msg.Status(value=rv_msg.Status.ERROR)


def serve(host="localhost:18300"):
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    rvs = RVizServer()
    rvs.initialize()
    rv_rpc.add_RVizServicer_to_server(rvs, server)
    server.add_insecure_port(host)
    server.start()
    print("The RViz server is started!")
    rvs.base.run()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve(host="192.168.1.111:182001")
