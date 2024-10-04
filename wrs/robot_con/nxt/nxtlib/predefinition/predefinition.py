import wrs.robot_con.nxt.nxtlib.nextage_client as nextage_client
from hrpsys import rtm

def pred():
    host = "nxtlib"
    rtm.nshost = host
    modelfile = "/opt/jsk/etc/NEXTAGE/model/main.wrl"
    robot = nextage_client.NextageClient()
    robot.init(robotname=host, url=modelfile)

    return robot