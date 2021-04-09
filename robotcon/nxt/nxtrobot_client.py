import grpc
import yaml
import nxtrobot_pb2 as nxt_msg
import nxtrobot_pb2_grpc as nxt_rpc

class NxtRobot(object):

    def __init__(self, host = "localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = nxt_rpc.NxtStub(channel)
        self._oldyaml = True
        if int(yaml.__version__[0]) >= 5:
            self._oldyaml = False

    def checkEncoders(self):
        returnvalue = self.stub.checkEncoders(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("Encoders succesfully checked.")

    def servoOn(self):
        returnvalue = self.stub.servoOn(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("Servos are turned on.")

    def servoOff(self):
        returnvalue = self.stub.servoOff(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("Servos are turned off.")

    def goInitial(self):
        returnvalue = self.stub.goInitial(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to its initial pose.")

    def goOffPose(self):
        returnvalue = self.stub.goOffPose(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the off pose.")

    def getJointAngles(self):
        if self._oldyaml:
            jntangles = yaml.load(self.stub.getJointAngles(nxt_msg.Empty()).data)
        else:
            jntangles = yaml.load(self.stub.getJointAngles(nxt_msg.Empty()).data, Loader=yaml.UnsafeLoader)
        return jntangles

    def setJointAngles(self, angles, tm = None):
        """
        All angles are in degree
        The tm is in second
        :param angles: [degree]
        :param tm: None by default
        :return:
        author: weiwei
        date: 20190417
        """
        returnvalue = self.stub.setJointAngles(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")

    def playPattern(self, angleslist, tmlist = None):
        """
        :param angleslist: [[degree]]
        :param tm: [second]
        :return:
        author: weiwei
        date: 20190417
        """
        if tmlist is None:
            tmlist = [.3]*len(angleslist)
        returnvalue = self.stub.playPattern(nxt_msg.SendValue(data = yaml.dump([angleslist, tmlist]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has finished the given motion.")

    def closeHandToolRgt(self):
        returnvalue = self.stub.closeHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has closed its right handtool.")

    def closeHandToolLft(self):
        returnvalue = self.stub.closeHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has closed its left handtool.")

    def openHandToolRgt(self):
        returnvalue = self.stub.openHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has opened its right handtool.")

    def openHandToolLft(self):
        returnvalue = self.stub.openHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has opened its left handtool.")

    def attachHandToolRgt(self):
        returnvalue = self.stub.attachHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has attached its right handtool.")

    def attachHandToolLft(self):
        returnvalue = self.stub.attachHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has attached its left handtool.")

    def ejectHandToolRgt(self):
        returnvalue = self.stub.ejectHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has ejected its right handtool.")

    def ejectHandToolLft(self):
        returnvalue = self.stub.ejectHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has ejected its left handtool.")


if __name__ == "__main__":
    nxt = NxtRobot(host = "10.0.1.102:18300")
    # nxt.servoOff()
    # nxt.servoOn()
    nxt.checkEncoders()
    nxt.goInitial()
    angles=[0,0,0,-15,0,-143,0,0,0,15,0,-143,0,0,0]
    import math
    anglesrad=[]
    for angle in angles:
        anglesrad.append(math.radians(angle))
    print(anglesrad)
    # nxt.playPattern([anglesrad], [5.0])
    # nxt.goOffPose()