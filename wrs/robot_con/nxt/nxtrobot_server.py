import grpc
import time
import yaml
import math
from concurrent import futures
import nxtrobot_pb2 as nxt_msg
import nxtrobot_pb2_grpc as nxt_rpc
import nxtlib.predefinition.predefinition as pre_def

class NxtServer(nxt_rpc.NxtServicer):
    """
    NOTE: All joint angle parameters are in degrees
    """

    _groups = [['torso', ['CHEST_JOINT0']],
               ['head', ['HEAD_JOINT0', 'HEAD_JOINT1']],
               ['rarm', ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2',
                         'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']],
               ['larm', ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2',
                         'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5']]]
    _initpose = [0,0,0,-15,0,-143,0,0,0,15,0,-143,0,0,0]
    _offpose = OffPose = [0,0,0,25,-140,-150,45,0,0,-25,-140,-150,-45,0,0]

    def _deg2rad(self, degreelist):
        return list(map(math.radians, degreelist))

    def _rad2deg(self, radianlist):
        return list(map(math.degrees, radianlist))

    def initialize(self):
        """
        MUST configure the robot_s in the very beginning
        :return:
        author: weiwei
        date: 20190417
        """
        self._robot = pre_def.pred()
        self._oldyaml = True
        if int(yaml.__version__[0]) >= 5:
            self._oldyaml = False

    def checkEncoders(self, request, context):
        try:
            self._robot.checkEncoders()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def servoOn(self, request, context):
        try:
            self._robot.servoOn()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def servoOff(self, request, context):
        try:
            self._robot.servoOff()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def goInitial(self, request, context):
        try:
            self._robot.goInitial()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def goOffPose(self, request, context):
        try:
            self._robot.goOffPose()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def getJointAngles(self, request, context):
        jntangles = self._robot.getJointAngles()
        return nxt_msg.ReturnValue(data = yaml.dump(jntangles))

    def setJointAngles(self, request, context):
        """
        :param request: request.data is in degree
        :param context:
        :return:
        author: weiwei
        date: 20190419
        """
        try:
            if self._oldyaml:
                angles, tm = yaml.load(request.data)
            else:
                angles, tm = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            if tm is None:
                tm = 10.0
            self._robot.playPattern([self._deg2rad(angles), [tm]])
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def playPattern(self, request, context):
        try:
            if self._oldyaml:
                angleslist, tmlist = yaml.load(request.data)
            else:
                angleslist, tmlist = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            self._robot.playPattern(angleslist, [], [], tmlist)
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def closeHandToolRgt(self, request, context):
        try:
            self._robot.gripper_r_close()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def closeHandToolLft(self, request, context):
        try:
            self._robot.gripper_l_close()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def openHandToolRgt(self, request, context):
        try:
            self._robot.gripper_r_open()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def openHandToolLft(self, request, context):
        try:
            self._robot.gripper_l_open()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def attachHandToolRgt(self, request, context):
        try:
            self._robot.handtool_r_attach()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def attachHandToolLft(self, request, context):
        try:
            self._robot.handtool_l_attach()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def ejectHandToolRgt(self, request, context):
        try:
            self._robot.handtool_r_eject()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def ejectHandToolLft(self, request, context):
        try:
            self._robot.handtool_l_eject()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

def serve():
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nxtserver = NxtServer()
    nxtserver.initialize()
    nxt_rpc.add_NxtServicer_to_server(nxtserver, server)
    server.add_insecure_port('[::]:18300')
    server.start()
    print("The Nextage Robot server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()