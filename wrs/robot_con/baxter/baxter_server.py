"""
AUTHOR: CHEN HAO
"""
import baxter_server_pb2 as bxtsp
import baxter_server_pb2_grpc as bxtspgc
import grpc
import time
from concurrent import futures
import roslayer
import pickle
import numpy as np

###################################
# import !
# Add a system environment into pycharm when you run it
# Run >> Edit Configurations >> Environment Varaibles >> add following variable
# LD_LIBRARY_PATH=/opt/ros/indigo/lib
# otherwise you cannot use the camera to fetch the image
########################

class BaxterServer(bxtspgc.BaxterServerServicer):

    def __init__(self):
        self.baxter = roslayer.Baxter()
        self.jnt_names = ["_s0","_s1","_e0","_e1","_w0","_w1","_w2"]

    def bxt_set_gripper(self, request, context):
        pos = request.pos
        armname = request.armname
        if pos > 99.9:
            self.baxter.opengripper(armname=armname)
        elif pos < 0.1:
            self.baxter.closegripper(armname=armname)
        else:
            self.baxter.commandgripper(pos=pos,armname=armname)
        return bxtsp.Empty()

    def bxt_get_gripper(self, request, context):
        armname = request.armname
        pos = self.baxter.currentposgripper(armname=armname)
        return bxtsp.Gripper_pos(pos=pos)

    def bxt_get_jnts(self,request, context):
        armname = request.armname
        jnts = self.baxter.getjnts(armname=armname)
        return bxtsp.Jnt_angles(jnt_angles=pickle.dumps(jnts))

    def bxt_movejnts(self,request, context):
        armname = request.armname
        speed = request.speed
        jnt_angles_r = np.frombuffer(request.jnt_angles).tolist()
        fullarmname = "right" if armname == "rgt" else "left"
        jnt_names = self.jnt_names
        jnt_angles = {}
        map(lambda name,jnts:jnt_angles.setdefault(fullarmname+name,np.deg2rad(jnts)),jnt_names,jnt_angles_r)
        self.baxter.movejnts(jnt_angles,speed,armname)
        return bxtsp.Empty()

    def bxt_movejnts_cont(self,request, context):
        armname = request.armname
        speed = request.speed
        jnt_angles_list_r = np.frombuffer(request.jnt_angles).reshape(-1,7)
        fullarmname = "right" if armname == "rgt" else "left"
        jnt_angles_list = []
        jnt_names = self.jnt_names
        for jnts_l in jnt_angles_list_r:
            jnts_dict = {}
            map(lambda name,jnts:jnts_dict.setdefault(fullarmname+name,np.deg2rad(jnts)),jnt_names,jnts_l.tolist())
            jnt_angles_list.append(jnts_dict)
        self.baxter.movejnts_cont(jnt_angles_list, speed, armname)
        return bxtsp.Empty()

    def bxt_get_force(self, request, context):
        armname = request.armname
        force = self.baxter.getforce(armname=armname)
        return bxtsp.ListData(list=np.array(force).tostring())

    def bxt_get_image(self, request, context):
        camera_name = request.name
        image = self.baxter.getimage(camera_name)
        return bxtsp.ListData(list=image.tostring())

def serve():
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options =[
        ("grpc.max_send_message_length",900*1000*1024),("grpc.max_receive_message_length",900*1000*1024)])
    bxtspgc.add_BaxterServerServicer_to_server(BaxterServer(), server)
    server.add_insecure_port('0.0.0.0:18300')
    server.start()
    print("The Baxter server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
