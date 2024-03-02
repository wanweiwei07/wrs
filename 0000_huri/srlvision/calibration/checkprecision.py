from commonimport import *

if __name__ == '__main__':
    yhx = robothelper.RobotHelperX(usereal=True, startworld=True)

    yhx.pxc.triggerframe()
    realpcd = yhx.pxc.getpcd()
    sensorhomomat = pickle.load(open(os.path.join(yhx.path, "datacalibration", "calibmat.pkl"), "rb"))
    pcd = rm.homotransformpointarray(sensorhomomat, realpcd)
    pcdnp = yhx.p3dh.genpointcloudnodepath(pcd)
    pcdnp.reparentTo(yhx.base.render)

    # refinedsensorhomomat = pickle.load(open(os.path.join(yhx.path, "datacalibration", "refinedcalibmat.pkl"), "rb"))
    # refinedpcd = rm.homotransformpointarray(refinedsensorhomomat, realpcd)
    # refinedpcdnp = yhx.p3dh.genpointcloudnodepath(refinedpcd, colors=[.5,1,.5,1])
    # refinedpcdnp.reparentTo(yhx.base.render)

    # yhx.robot_s.movearmfk(yhx.rbtx.getarmjntsx(arm_name="rgt"), arm_name="rgt")
    # tcppos, tcprot = yhx.robot_s.gettcp()
    # yhx.p3dh.genframe(tcppos, tcprot, major_radius=15).reparentTo(yhx.base.render)
    # minx = tcppos[0]-100
    # maxx = minx+100
    # miny = tcppos[1]
    # maxy = miny+140
    # minz = tcppos[2]
    # maxz = tcppos[2]+70
    # mph = o3dh.cropnx3nparray(mph, [minx, maxx], [miny, maxy], [minz, maxz])

    yhx.show()