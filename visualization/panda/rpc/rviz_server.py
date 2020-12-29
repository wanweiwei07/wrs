import grpc
import time
import numpy as np
from concurrent import futures
import visualization.panda.rpc.rviz_pb2 as rv_msg
import visualization.panda.rpc.rviz_pb2_grpc as rv_rpc
import visualization.panda.world as wd


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
            exec(code, globals())
            return rv_msg.Status(value = rv_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return rv_msg.Status(value = rv_msg.Status.ERROR)

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
