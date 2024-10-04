import grpc
import pickle
from concurrent import futures
import wrs.modeling.model_collection as mc
import wrs.visualization.panda.rpc.rviz_pb2 as rv_msg
import wrs.visualization.panda.rpc.rviz_pb2_grpc as rv_rpc
import wrs.visualization.panda.world as wd
from wrs import robot_sim as ri


class RVizServer(rv_rpc.RVizServicer):

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

    def create_instance(self, request, context):
        """
        :param request:
        :param context:
        :return:
        author: weiwei
        date: 20201231
        """
        try:
            name = request.name
            data = request.data
            globals()[name] = pickle.loads(data)
            # fix the unserializable Shaders and CollisionTraversers
            # https://discourse.panda3d.org/t/serializing-pandanode-shaders-collisiontraverser-etc/26945/5
            # https://github.com/panda3d/panda3d/issues/1090
            if isinstance(globals()[name], gm.GeometricModel):
                globals()[name].pdndp_core.setShaderAuto()
            elif isinstance(globals()[name], mc.ModelCollection):
                for cm in globals()[name].cm_list:
                    cm.pdndp_core.setShaderAuto()
                for gm in globals()[name].gm_list:
                    if isinstance(gm, gm.GeometricModel):
                        gm.pdndp_core.setShaderAuto()
            elif isinstance(globals()[name], ri.RobotInterface):
                globals()[name].enable_cc()
            return rv_msg.Status(value=rv_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return rv_msg.Status(value=rv_msg.Status.ERROR)

def serve(host="localhost:18300"):
    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_send_message_length', 100 * 1024 * 1024),
               ('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    rvs = RVizServer()
    rv_rpc.add_RVizServicer_to_server(rvs, server)
    server.add_insecure_port(host)
    server.start()
    print("The RViz server is started!")
    base.run()


if __name__ == "__main__":
    serve(host="localhost:182001")
