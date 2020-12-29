import robotconn.rpc.nxtrobot.nxtrobot_client as nxtc

if __name__ == "__main__":
    nxt = nxtc.NxtRobot(host="10.0.1.102:18300")
    nxt.servoOff()
    nxt.servoOn()