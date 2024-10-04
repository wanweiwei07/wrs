import tro.tro_robothelper as robothelper
import numpy as np
import utiltools.robotmath as rm
import environment.collisionmodel as cm

if __name__ == '__main__':

    yhx = robothelper.RobotHelperX(usereal=True)
    yhx.movetox(yhx.robot_s.initrgtjnts, armname="rgt")
    yhx.movetox(yhx.robot_s.initlftjnts, armname="lft")
    yhx.closegripperx(armname="rgt")
    # yhx.closegripperx(arm_name="lft")
    # yhx.opengripperx(arm_name="rgt")
    # yhx.opengripperx(arm_name="lft")


    objcm = cm.CollisionModel("../objects/vacuumhead.stl")

    # yhx = robothelper.RobotHelper(startworld=True)
    armjnts0 = np.array([120.18, -41.71, -160.45, 7.16, 148.59, 95.69, -210.39])
    obstaclecmlist = []
    armname = "rgt"
    rgtinitarmjnts = yhx.robot_s.getarmjnts(armname=armname).tolist()
    primitivedirection_backward = np.array([0,0,1])
    primitivedistance_backward = 150
    rgtupmotion = yhx.genmovebackwardmotion(primitivedirection_backward, primitivedistance_backward,
                                         armjnts0, obstaclecmlist, armname)
    rgtpickmotion = rgtupmotion[::-1]
    rgtrrtgotobeforepick = yhx.planmotion(rgtinitarmjnts, rgtpickmotion[0], obstaclecmlist, armname)
    armname = "lft"
    lftinitarmjnts = yhx.robot_s.getarmjnts(armname=armname).tolist()
    fakelftupmotion = [lftinitarmjnts]*len(rgtupmotion)
    fakelftpickmotion = [lftinitarmjnts]*len(rgtpickmotion)
    fakelftrrtgotobeforepick = [lftinitarmjnts]*len(rgtrrtgotobeforepick)

    rgtmotion = rgtrrtgotobeforepick+rgtpickmotion+rgtupmotion
    lftmotion = fakelftrrtgotobeforepick+fakelftpickmotion+fakelftupmotion

    yhx.movemotionx(rgtrrtgotobeforepick+rgtpickmotion, armname='rgt')
    yhx.toggleonsuction(armname='rgt')
    yhx.movemotionx(rgtupmotion, armname='rgt')

    armjnts1 = np.array([114.02, -5.76, -123.35, 40.58, 229.27, 75.28, -227.12])
    obstaclecmlist = []
    armname = "rgt"
    rgtinitarmjnts = rgtupmotion[-1]
    yhx.robot_s.movearmfk(armjnts1, armname)
    # primitivedirection_backward = -yhx.robot_s.getee(arm_name)[1][:,0]
    primitivedirection_backward = np.array([0,-1,1])
    primitivedistance_backward = 150
    rgtbackmotion = yhx.genmovebackwardmotion(primitivedirection_backward, primitivedistance_backward,
                                         armjnts1, obstaclecmlist, armname)
    rgtrrtgotobeforehandover = yhx.planmotion(rgtinitarmjnts, rgtbackmotion[0], obstaclecmlist, armname)
    fakelftrrtgotobeforehandover = [lftinitarmjnts]*len(rgtrrtgotobeforehandover)

    rgtmotion = rgtmotion+rgtrrtgotobeforehandover
    lftmotion = lftmotion+fakelftrrtgotobeforehandover

    yhx.movemotionx(rgtrrtgotobeforehandover, armname='rgt')

    armjnts2 = np.array([-66.7, -90.01, 77.63, 49.53, 136.47, 32.54, -206.18])
    armname = "lft"
    rgtinitarmjnts = rgtrrtgotobeforehandover[-1]
    yhx.robot_s.movearmfk(armjnts2, armname)
    primitivedirection_backward = -yhx.robot_s.getee(armname)[1][:, 2]
    primitivedistance_backward = 100
    lftbackmotion = yhx.genmovebackwardmotion(primitivedirection_backward, primitivedistance_backward,
                                            armjnts2, obstaclecmlist, armname)
    lftforwardmotion = lftbackmotion[::-1]
    lftrrtgotobeforehandoverreceive = yhx.planmotion(lftinitarmjnts, lftforwardmotion[0], obstaclecmlist, armname)
    fakergtrrtgotobeforehandoverreceive = [rgtinitarmjnts]*len(lftrrtgotobeforehandoverreceive)
    fakergtforwardmotion = [rgtinitarmjnts]*len(lftforwardmotion)

    rgtmotion = rgtmotion+fakergtrrtgotobeforehandoverreceive+fakergtforwardmotion
    lftmotion = lftmotion+lftrrtgotobeforehandoverreceive+lftforwardmotion
    #
    yhx.movemotionx(lftrrtgotobeforehandoverreceive, armname='lft')
    yhx.movemotionx(lftforwardmotion, armname='lft')
    yhx.closegripperx(armname='lft')

    fakelftbackmotion = [lftforwardmotion[-1]]*len(rgtbackmotion)

    rgtmotion = rgtmotion+rgtbackmotion
    lftmotion = lftmotion+fakelftbackmotion
    #
    yhx.toggleoffsuction(armname='rgt')
    yhx.movemotionx(rgtbackmotion, armname='rgt')

    # armjnts3 = np.array([-80.67, -89.99, 68.17, 45.50, -25.60, -87.97, -170.08])
    # armjnts3 = np.array([-63.77, -89.99, 63.95, 40.06, -28.76, -66, -149.78])
    # armjnts3 = np.array([-63.25, -91, 66.09, 44.74, -31.33, -71.22, -150.79])
    # armjnts3 = np.array([-55.57, -66.58, 80.62, 18.66, -205.87, -17.62, -124.98])
    armjnts3 = np.array([-61.65, -58.98, 90.48, 12.00, -234.39, -31.60, -101.99])
    armname = "lft"
    lftinitarmjnts = lftforwardmotion[-1] # yhx.robot_s.movearmfk(armjnts3, arm_name)
    primitivedirection_forward = np.array([0,0,1])
    primitivedistance_forward = -200
    lftforwardmotion = yhx.genmoveforwardmotion(primitivedirection_forward, primitivedistance_forward,
                                         armjnts3, obstaclecmlist, armname)
    lftdownwardmotion = lftforwardmotion
    yhx.robot_s.movearmfk(armjnts3, armname)
    yhx.robot_s.movearmfk(rgtbackmotion[-1], "rgt")
    lftrrtgotobeforeinsertion = yhx.planmotionhold(lftinitarmjnts, lftdownwardmotion[0], objcm=objcm, relpos=np.array([0,0,0]), relrot=rm.rodrigues(np.array([0,0,1]), 90), obscmlist=obstaclecmlist, armname=armname)

    fakergtrrtgotobeforeinsertion = [rgtbackmotion[-1]]*len(lftrrtgotobeforeinsertion)
    fakergtdownwardmotion = [rgtbackmotion[-1]]*len(lftdownwardmotion)

    rgtmotion = rgtmotion+fakergtrrtgotobeforeinsertion+fakergtdownwardmotion
    lftmotion = lftmotion+lftrrtgotobeforeinsertion+lftdownwardmotion

    #
    yhx.movemotionx(lftrrtgotobeforeinsertion, armname='lft')
    yhx.movemotionx(lftdownwardmotion, armname='lft')

    rbtmnp = [None]
    motioncounter = [0]
    def render(yh, rgtmotion, lftmotion, rbtmnp, motioncounter, task):
        print(motioncounter[0])
        if motioncounter[0] < len(rgtmotion):
            if rbtmnp[0] is not None:
                rbtmnp[0].detachNode()
            yh.robot_s.movealljnts([0, 0, 0] + rgtmotion[motioncounter[0]] + lftmotion[motioncounter[0]])
            rbtmnp[0] = yh.rbtmesh.genmnp(yh.robot_s)
            rbtmnp[0].reparentTo(yh.base.render)
            motioncounter[0] += 1
        else:
            motioncounter[0] = 0
        return task.again
    taskMgr.doMethodLater(0.01, render, "render",
                          extraArgs=[yhx, rgtmotion, lftmotion, rbtmnp, motioncounter],
                          appendTask=True)
    yhx.base.run()
