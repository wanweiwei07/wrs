import robotconn.rpc.baxterrobot.baxter_server_pb2 as bxtsp
import robotconn.rpc.baxterrobot.baxter_server_pb2_grpc as bxtspgc
import grpc
import pickle
import numpy as np

class BaxterClient(object):

    def __init__(self, host = "localhost:18300"):

        channel = grpc.insecure_channel(host)
        self.stub = bxtspgc.BaxterServerStub(channel)

    def bxt_set_gripper(self, pos=100, armname = "rgt"):
        self.stub.bxt_set_gripper(bxtsp.Gripper_pos_armname(pos=pos,armname=armname))

    def bxt_get_gripper(self, armname="rgt"):
        return self.stub.bxt_get_gripper(bxtsp.Armname(armname=armname))

    def bxt_get_jnts(self, armname="rgt"):
        jnts =  pickle.loads(self.stub.bxt_get_jnts(bxtsp.Armname(armname=armname)).jnt_angles)
        jnts = [jnts["right_s0"],jnts["right_s1"],jnts["right_e0"],jnts["right_e1"],jnts["right_w0"],jnts["right_w1"],jnts["right_w2"]] \
            if armname == "rgt" else [jnts["left_s0"],jnts["left_s1"],jnts["left_e0"],jnts["left_e1"],jnts["left_w0"],jnts["left_w1"],jnts["left_w2"]]
        jnts = [np.rad2deg(jnt) for jnt in jnts]
        return jnts

    def bxt_movejnts(self, jnt_angles= [], speed=.5, armname="rgt"):
        self.stub.bxt_movejnts(bxtsp.Jnt_angles_armname(jnt_angles = np.array(jnt_angles,dtype="float").tobytes(),speed=speed,armname =armname))

    def bxt_movejnts_cont(self, jnt_angles_list =[], speed=.2, armname="rgt"):
        self.stub.bxt_movejnts_cont(bxtsp.Jnt_angles_armname(jnt_angles = np.array(jnt_angles_list,dtype="float").tobytes(),speed=speed,armname =armname))

    def bxt_get_force(self,armname):
        return np.frombuffer(self.stub.bxt_get_force(bxtsp.Armname(armname=armname)).list).tolist()

    def bxt_get_image(self,camera_name):
        image = self.stub.bxt_get_image(bxtsp.Camera_name(name=camera_name)).list
        image = np.frombuffer(image)
        image = np.reshape(image,(200,320,3)).astype("uint8")
        # image = image[:,:,1]
        return image

if __name__=="__main__":
    import time

    bc = BaxterClient(host = "10.1.0.24:18300")
    # tic = time.time()
    # imgx = hcc.getimgbytes()
    # toc = time.time()
    # td = toc-tic
    # tic = time.time()
    # imgxs = hcc.getimgstr()
    # toc = time.time()
    # td2 = toc-tic
    # print(td, td2)
    angle_rgt = bc.bxt_get_jnts("rgt")
    # print angle_rgt
    # print(angle_rgt[-1])
    #
    #
    # angle_rgt[-1] = angle_rgt[-1] - 50.0
    #
    # bc.bxt_movejnts(angle_rgt)
    print(bc.bxt_get_jnts(armname="rgt"))
    print(bc.bxt_get_jnts(armname="lft"))
    import cv2 as cv
    cv.imshow("w",bc.bxt_get_image("head_camera"))
    cv.waitKey(0)
    # print  bc.bxt_get_jnts("rgt")
    # print(eval("a="+bc.bxt_get_jnts()))