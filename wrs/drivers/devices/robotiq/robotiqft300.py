import logging


class RobotiqFT300Sensor(object):
    complete_program = ""
    header = "def myProg():" + "\n"
    end = "\n" + "end_type"
    logger = False

    def __init__(self):
        self.logger = logging.getLogger("urx")
        self.reset()

    def add_line_to_program(self, new_line):
        if (self.complete_program != ""):
            self.complete_program += "\n"
        self.complete_program += new_line

    def reset(self):
        self.complete_program = ""
        self.add_line_to_program("socket_close(\"ftsensor_socket\")")
        # self.add_line_to_program("sleep(1)") #in Robotiq's example they do a wait here... I haven't found it neccessary
        self.add_line_to_program("socket_open(\"127.0.0.1\",63351,\"ftsensor_socket\")")
        # self.add_line_to_program("sleep(1)")
        self.add_line_to_program("write_port_register(410, 0x200)")
        self.add_line_to_program("sync()")

    # def receiveftdata(self):
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     # now connect to the web server on port 80 - the normal http port
    #     s.connect(("10.2.0.50", 63351))
    #     while True:
    #         print s.recv(1024)

    def ret_program_to_run(self):
        if (self.complete_program == ""):
            self.logger.debug("ftsensor's program is empty")
            return ""

        prog = self.header
        prog += self.complete_program
        prog += self.end
        return prog


# if __name__=="__main__":
#
#     ur3u = ur3urx.Ur3DualUrx()
#     ft300u = Robotiq_FT300_Sensor()
#     ur3u.rgtarm.send_program(ft300u.ret_program_to_run())
#     ft300u.receiveftdata()


if __name__ == '__main__':
    import pandaplotutils.pandactrl as pandactrl
    from wrs import robot_sim as ur3dual, robot_sim as ur3dualmesh, robot_sim as ur3dualball, manipulation as rtq85

    import robotconn.ur3dual as ur3urx

    base = pandactrl.World()

    robot = ur3dual.Ur3DualRobot()
    robot.goinitpose()
    rgthnd = rtq85.Rtq85()
    lfthnd = rtq85.Rtq85()
    robotmesh = ur3dualmesh.Ur3DualMesh(rgthand=rgthnd, lfthand=lfthnd)
    robotball = ur3dualball.Ur3DualBall()
    ur3u = ur3urx.Ur3DualUrx()
    # cdchecker = cdck.CollisionChecker(robotmesh)
    cdchecker = cdck.CollisionCheckerBall(robotball)

    start = ur3u.get_jnt_values('rgt')
    goal = robot.initjnts[3:9]
    # start = [50.0,0.0,-143.0,0.0,0.0,0.0]
    # goal = [-15.0,0.0,-143.0,0.0,0.0,0.0]
    # plot init and goal
    robot.movearmfk(armjnts=start, armid='rgt')
    robotmesh.genmnp(robot).reparentTo(base.render)
    robot.movearmfk(armjnts=goal, armid='rgt')
    robotmesh.genmnp(robot).reparentTo(base.render)

    jointlimits = [[robot.rgtarm[1]['rngmin'], robot.rgtarm[1]['rngmax']],
                   [robot.rgtarm[2]['rngmin'], robot.rgtarm[2]['rngmax']],
                   [robot.rgtarm[3]['rngmin'], robot.rgtarm[3]['rngmax']],
                   [robot.rgtarm[4]['rngmin'], robot.rgtarm[4]['rngmax']],
                   [robot.rgtarm[5]['rngmin'], robot.rgtarm[5]['rngmax']],
                   [robot.rgtarm[6]['rngmin'], robot.rgtarm[6]['rngmax']]]
    # import os
    # from panda3d.core import *
    # import pandaplotutils.pandageom as pg
    # obsrotmat4 = Mat4(1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,355.252044678,-150.073120117,200.0000038147,1.0)
    # this_dir, this_filename = os.path.split(__file__)
    # obj_path = os.path.join(os.path.split(this_dir)[0], "manipulation/grip", "objects", "tool.stl")
    # objmnp = pg.genObjmnp(obj_path, color=Vec4(.7, .7, 0, 1))
    # objmnp.setMat(obsrotmat4)
    # objmnp.reparentTo(base.render)

    # planner = rrt.RRT(start=start, goal=goal, iscollidedfunc = iscollidedfunc,
    #                   jointlimits = jointlimits, expanddis = 5,
    #                   robot_s = robot_s, cdchecker = cdchecker)
    # planner = rrtc.RRTConnect(start=start, goal=goal, iscollidedfunc = iscollidedfunc,
    #           jointlimits = jointlimits, expanddis = 10, robot_s = robot_s, cdchecker = cdchecker)
    # planner = ddrrt.DDRRT(start=start, goal=goal, iscollidedfunc = iscollidedfunc,
    #                   jointlimits = jointlimits, goalsamplerate=30, expanddis = 5, robot_s = robot_s,
    #                   cdchecker = cdchecker)
    #
    planner = ddrrtc.DDRRTConnect(start=start, goal=goal, iscollidedfunc=iscollidedfunc,
                                  jointlimits=jointlimits, starttreesamplerate=30, expanddis=5, robot=robot,
                                  cdchecker=cdchecker)

    import time

    tic = time.time()
    [path, sampledpoints] = planner.planning(obstaclelist=[])
    toc = time.time()
    print(toc - tic)
    #
    for pose in path:
        ur3u.movejntssgl_cont(pose, armid='rgt')
        robot.movearmfk(pose, armid='rgt')
        robotstick = robotmesh.gensnp(robot=robot)
        robotstick.reparentTo(base.render)
    ur3u.move_jnts(path[-1], armid='rgt')

    start = ur3u.get_jnt_values('lft')
    goal = robot.initjnts[9:15]
    print(start, goal)
    jointlimits = [[robot.lftarm[1]['rngmin'], robot.lftarm[1]['rngmax']],
                   [robot.lftarm[2]['rngmin'], robot.lftarm[2]['rngmax']],
                   [robot.lftarm[3]['rngmin'], robot.lftarm[3]['rngmax']],
                   [robot.lftarm[4]['rngmin'], robot.lftarm[4]['rngmax']],
                   [robot.lftarm[5]['rngmin'], robot.lftarm[5]['rngmax']],
                   [robot.lftarm[6]['rngmin'], robot.lftarm[6]['rngmax']]]
    planner = ddrrtc.DDRRTConnect(start=start, goal=goal, iscollidedfunc=iscollidedfunc,
                                  jointlimits=jointlimits, starttreesamplerate=30, expanddis=5, robot=robot,
                                  cdchecker=cdchecker)

    import time

    tic = time.time()
    [path, sampledpoints] = planner.planning(obstaclelist=[])
    toc = time.time()
    print(toc - tic)
    #
    for pose in path:
        ur3u.movejntssgl_cont(pose, armid='lft')
        robot.movearmfk(pose, armid='lft')
        robotstick = robotmesh.gensnp(robot=robot)
        robotstick.reparentTo(base.render)
    ur3u.move_jnts(path[-1], armid='lft')

    ur3u.close_gripper(armid='lft')
    ur3u.close_gripper(armid='rgt')

    while True:
        print(ur3u.recvft(armid='rgt')[0])
        # print ur3u.recvft(armid = 'lft')


    def getftthread(ur3u, armid='rgt'):
        print(ur3u.recvft(armid=armid))


    import threading

    thread = threading.Thread(target=getftthread, args=([ur3u, 'rgt']), name="threadft")
    thread.start()

    base.run()
