import grpc
from concurrent import futures
import wrs.visualization.panda.world as wd
import wrs.visualization.panda.rpc.rviz_server as rs
import wrs.visualization.panda.rpc.rviz_pb2_grpc as rv_rpc


def serve(host="localhost:18300"):
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
               ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    rvs = rs.RVizServer()
    rv_rpc.add_RVizServicer_to_server(rvs, server)
    server.add_insecure_port(host)
    server.start()
    print("The RViz server is started!")
    base.run()


if __name__ == "__main__":
    serve(host="localhost:182001")
