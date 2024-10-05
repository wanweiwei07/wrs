import wrs.drivers.dobot_magician.DobotDllType as dd

CON_STR = {
    dd.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dd.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dd.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}


class Dobot_X(object):

    def __init__(self, port):
        """
        :param port: must be one of your pc's com ports
        author: weiwei
        date: 20220215
        """
        self.api = dd.load()
        self.state = dd.ConnectDobot(self.api, port, 115200)[0]
        print("Connect status:", CON_STR[self.state])
        if (self.state == dd.DobotConnect.DobotConnect_NoError):
            print("Successfully connected to Dobot!")
            # Async Motion Params Setting
            dd.SetHOMEParams(self.api, 200, 200, 200, 200, isQueued=1)
            dd.SetPTPJointParams(self.api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1)
            dd.SetPTPCommonParams(self.api, 100, 100, isQueued=1)
            # Async Home
            # dd.SetHOMECmd(self.api, temp=0, isQueued=1)
        else:
            print("Connection Failed! Try Again")
            exit(0)

    def __del__(self):
        if (self.state == dd.DobotConnect.DobotConnect_NoError):
            dd.DisconnectDobot(self.api)
            print("Disconneted from Dobot!")

    def get_jnts(self):
        """
        joint values
        :return:
        """
        x, y, z, r, j1, j2, j3, j4 = dd.GetPose(self.api)
        return np.radians([j1, j2, j3, j4])

    def get_pose(self):
        """
        tcp pose
        :return:
        """
        x, y, z, r, j1, j2, j3, j4 = dd.GetPose(self.api)
        return x, y, z, r


if __name__ == '__main__':
    import numpy as np
    import wrs.robot_sim.manipulators.dobot_magician.dobot_magician as dm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])

    robot_s = dm.DobotMagician()
    robot_c = Dobot_X(port='com4')
    jnt_values = robot_c.get_jnts()
    print(jnt_values)
    robot_s.fk(jnt_values[:3])
    #
    robot_s.gen_meshmodel().attach_to(base)
    base.run()
