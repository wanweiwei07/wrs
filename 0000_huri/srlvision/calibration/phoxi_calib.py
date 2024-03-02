import robothelper
import numpy as np
from cv2 import aruco as aruco
import utiltools.robotmath as rm
import utiltools.thirdparty.o3dhelper as o3dh
from scipy.optimize import leastsq

def getcenter(img, pcd, aruco_dict, parameters, tgtids=[0,1]):
    """
    get the center of two markers

    :param img:
    :param pcd:
    :return:

    author: yuan gao, ruishuang, revised by weiwei
    date: 20161206
    """

    width = img.shape[1]
    # First, detect markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if len(corners) < len(tgtids):
        return None
    if len(ids) != len(tgtids):
        return None
    if ids[0] not in tgtids or ids[1] not in tgtids:
        return None
    center = np.mean(np.mean(corners, axis=0), axis=1)[0]
    center = np.array([int(center[0]), int(center[1])])
    # print(center)
    pos = pcd[width * center[1] + center[0]]

    return pos

def phoxi_computeeeinphx(yhx, pxc, armname, actionpos, actionrot, parameters, aruco_dict, criteriaradius=None):
    """

    :param yhx:
    :param pxc:
    :param armname:
    :param actionpos:
    :param actionrot:
    :param parameters:
    :param aruco_dict:
    :param criteriaradius: rough major_radius used to determine if the newly estimated center is correct or not
    :return:

    author: weiwei
    date: 20190110
    """

    def fitfunc(p, coords):
        x0, y0, z0, R = p
        x, y, z = coords.T
        return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
    errfunc = lambda p, x: fitfunc(p, x) - p[3]

    coords = []
    rangex = [np.array([1,0,0]), [-30,-15,0,15,30]]
    rangey = [np.array([0,1,0]), [-30,-15,15,30]]
    rangez = [np.array([0,0,1]), [-90,-60,-30,30,60]]
    rangeaxis = [rangex, rangey, rangez]
    lastarmjnts = yhx.robot_s.initrgtjnts
    for axisid in range(3):
        axis = rangeaxis[axisid][0]
        for angle in rangeaxis[axisid][1]:
            goalpos = actionpos
            goalrot = np.dot(rm.rodrigues(axis, angle), actionrot)
            armjnts = yhx.movetoposrotmsc(eepos=goalpos, eerot=goalrot, msc=lastarmjnts, armname=armname)
            if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
                lastarmjnts = armjnts
                yhx.movetox(armjnts, armname=armname)
                pxc.triggerframe()
                img = pxc.gettextureimg()
                pcd = pxc.getpcd()
                phxpos = getcenter(img, pcd, aruco_dict, parameters)
                if phxpos is not None:
                    coords.append(phxpos)
    print(coords)
    if len(coords) < 3:
        return [None, None]
    # for coord in coords:
    #     yhx.p3dh.gensphere(coord, major_radius=5).reparentTo(base.render)
    coords = np.asarray(coords)
    # try:
    initialguess = np.ones(4)
    initialguess[:3] = np.mean(coords, axis=0)
    finalestimate, flag = leastsq(errfunc, initialguess, args=(coords,))
    if len(finalestimate) == 0:
        return [None, None]
    print(finalestimate)
    print(np.linalg.norm(coords - finalestimate[:3], axis=1))
    # yhx.p3dh.gensphere(finalestimate[:3], rgba=np.array([0,1,0,1]), major_radius=5).reparentTo(base.render)
    # yhx.base.run()
    # except:
    #     return [None, None]
    if criteriaradius is not None:
        if abs(finalestimate[3]-criteriaradius) > 5:
            return [None, None]
    return np.array(finalestimate[:3]), finalestimate[3]

def phoxi_computeboardcenterinhand(yhx, pxc, armname, parameters, aruco_dict, criteriaradius=None):
    """

    :param yhx:
    :param pxc:
    :param armname:
    :param parameters:
    :param aruco_dict:
    :param criteriaradius: rough major_radius used to determine if the newly estimated center is correct or not
    :return:

    author: weiwei
    date: 20190110
    """

    actionpos = np.array([300,-50,200])
    actionrot = np.array([[0,0,1],[1,0,0],[0,1,0]]).T
    eeposinphx, bcradius = phoxi_computeeeinphx(yhx, pxc, armname, actionpos, actionrot, parameters, aruco_dict)

    # moveback
    armjnts = yhx.movetoposrot(eepos=actionpos, eerot=actionrot, armname=armname)
    if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
        yhx.movetox(armjnts, armname=armname)
        pxc.triggerframe()
        img = pxc.gettextureimg()
        pcd = pxc.getpcd()
        hcinphx = getcenter(img, pcd, aruco_dict, parameters)
    print(hcinphx)

    movedist = 100
    actionpos_hx = actionpos+actionrot[:,0]*movedist
    armjnts = yhx.movetoposrot(eepos=actionpos_hx, eerot=actionrot, armname=armname)
    if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
        yhx.movetox(armjnts, armname=armname)
        pxc.triggerframe()
        img = pxc.gettextureimg()
        pcd = pxc.getpcd()
        hxinphx = getcenter(img, pcd, aruco_dict, parameters)
    print(hxinphx)

    movedist = 100
    actionpos_hy = actionpos+actionrot[:,1]*movedist
    armjnts = yhx.movetoposrot(eepos=actionpos_hy, eerot=actionrot, armname=armname)
    if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
        yhx.movetox(armjnts, armname=armname)
        pxc.triggerframe()
        img = pxc.gettextureimg()
        pcd = pxc.getpcd()
        hyinphx = getcenter(img, pcd, aruco_dict, parameters)
    print(hyinphx)

    movedist = 100
    actionpos_hz = actionpos+actionrot[:,2]*movedist
    armjnts = yhx.movetoposrot(eepos=actionpos_hz, eerot=actionrot, armname=armname)
    if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
        yhx.movetox(armjnts, armname=armname)
        pxc.triggerframe()
        img = pxc.gettextureimg()
        pcd = pxc.getpcd()
        hzinphx = getcenter(img, pcd, aruco_dict, parameters)
    print(hzinphx)

    frameinphx = np.array([hxinphx-hcinphx, hyinphx-hcinphx, hzinphx-hcinphx]).T
    frameinphx, r = np.linalg.qr(frameinphx)
    bcinhnd = np.dot(frameinphx.T, hcinphx-eeposinphx)
    print(bcinhnd)


def phoxi_calibbyestinee(yhx, pxc, armname, parameters, aruco_dict):
    """

    :param yhx: an instancde of YumiHelperX
    :param pxc: phoxi client
    :param armname:
    :return:

    author: weiwei
    date: 20191228
    """

    realposlist = []
    phxposlist = []

    actionpos = np.array([350, 0, 200])
    actionrot = np.array([[0,0,1],[1,0,0],[0,1,0]]).T
    # estimate a criteriaradius
    phxpos, criteriaradius = phoxi_computeeeinphx(yhx, pxc, armname, actionpos, actionrot, parameters, aruco_dict)
    print(phxpos, criteriaradius)
    realposlist.append(actionpos)
    phxposlist.append(phxpos)
    print(phxposlist)

    for x in [250, 400]:
        for y in [-180, 180]:
            for z in [150, 250]:
                actionpos = np.array([x, y, z])
                phxpos, _ = phoxi_computeeeinphx(yhx, pxc, armname, actionpos, actionrot, parameters, aruco_dict, criteriaradius)
                if phxpos is not None:
                    realposlist.append(np.array([x, y, z]))
                    phxposlist.append(phxpos)
                    print(phxposlist)

    realposarr = np.asarray(realposlist)
    phxposarr = np.asarray(phxposlist)
    print(phxposarr)
    print(realposarr)
    amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
    pickle.dump(realposarr, open(os.path.join(yhx.path, "datacalibration", "realpos.pkl"), "wb"))
    pickle.dump(phxposarr, open(os.path.join(yhx.path, "datacalibration", "ampos.pkl"), "wb"))
    pickle.dump(amat, open(os.path.join(yhx.path, "datacalibration", "calibmat.pkl"), "wb"))
    print(amat)
    return amat

def phoxi_calib(yhx, pxc, armname, relpos, parameters, aruco_dict):
    """

    :param yhx: an instancde of YumiHelperX
    :param pxc: phoxi client
    :param relpos: relative pos in the hand frame
    :param armname:
    :return:

    author: weiwei
    date: 20191228
    """

    realposlist = []
    phxposlist = []

    lastarmjnts = yhx.robot_s.initrgtjnts
    eerot = np.array([[0,0,1],[1,0,0],[0,1,0]]).T # horizontal, facing right
    for x in [300, 360, 420]:
        for y in range(-200, 201, 200):
            for z in [70, 90, 130, 200]:
                armjnts = yhx.movetoposrotmsc(eepos=np.array([x, y, z]), eerot=eerot, msc=lastarmjnts, armname=armname)
                if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
                    lastarmjnts = armjnts
                    yhx.movetox(armjnts, armname=armname)
                    pxc.triggerframe()
                    img = pxc.gettextureimg()
                    pcd = pxc.getpcd()
                    phxpos = getcenter(img, pcd, aruco_dict, parameters)
                    print(phxpos)
                    if phxpos is not None:
                        realposlist.append(np.array([x, y, z]) + np.dot(eerot, relpos))
                        phxposlist.append(phxpos)

    realposarr = np.array(realposlist)
    phxposarr = np.array(phxposlist)
    amat = rm.affine_matrix_from_points(phxposarr.T, realposarr.T)
    pickle.dump(realposarr, open(os.path.join(yhx.path, "datacalibration", "realpos.pkl"), "wb"))
    pickle.dump(phxposarr, open(os.path.join(yhx.path, "datacalibration", "ampos.pkl"), "wb"))
    pickle.dump(amat, open(os.path.join(yhx.path, "datacalibration", "calibmat.pkl"), "wb"))
    print(amat)
    return amat

def phoxi_calib_refinewithmodel(yhx, pxc, rawamat, armname):
    """
    The performance of this refining method using cad model is not good.
    The reason is probably a precise mobdel is needed.

    :param yhx: an instancde of YumiHelperX
    :param pxc: phoxi client
    :param armname:
    :return:

    author: weiwei
    date: 20191228
    """

    handpalmtemplate = pickle.load(open(os.path.join(yhx.path, "dataobjtemplate", "handpalmtemplatepcd.pkl"), "rb"))

    newhomomatlist = []

    lastarmjnts = yhx.robot_s.initrgtjnts
    eerot = np.array([[1,0,0],[0,0,-1],[0,1,0]]).T # horizontal, facing right
    for x in [300, 360, 420]:
        for y in range(-200, 201, 200):
            for z in [70, 90, 130, 200]:
                armjnts = yhx.movetoposrotmsc(eepos=np.array([x, y, z]), eerot=eerot, msc=lastarmjnts, armname=armname)
                if armjnts is not None and not yhx.pcdchecker.isSelfCollided(yhx.robot_s):
                    lastarmjnts = armjnts
                    yhx.movetox(armjnts, armname=armname)
                    tcppos, tcprot = yhx.robot_s.gettcp()
                    initpos = tcppos+tcprot[:,2]*7
                    initrot = tcprot
                    inithomomat = rm.homobuild(initpos, initrot)
                    pxc.triggerframe()
                    pcd = pxc.getpcd()
                    realpcd = rm.homotransformpointarray(rawamat, pcd)
                    minx = tcppos[0]-100
                    maxx = tcppos[0]+100
                    miny = tcppos[1]
                    maxy = tcppos[1]+140
                    minz = tcppos[2]
                    maxz = tcppos[2]+70
                    realpcdcrop = o3dh.crop_nx3_nparray(realpcd, [minx, maxx], [miny, maxy], [minz, maxz])
                    if len(realpcdcrop) < len(handpalmtemplate)/2:
                        continue
                    # yhx.rbtmesh.genmnp(yhx.robot_s).reparentTo(base.render)
                    # yhx.p3dh.genframe(tcppos, tcprot, major_radius=10). reparentTo(base.render)
                    # yhx.p3dh.gensphere([minx, miny, minz], major_radius=10).reparentTo(base.render)
                    # yhx.p3dh.gensphere([maxx, maxy, maxz], major_radius=10).reparentTo(base.render)
                    # yhx.p3dh.genpointcloudnodepath(realpcd).reparentTo(base.render)
                    # yhx.p3dh.genpointcloudnodepath(realpcdcrop, colors=[1,1,0,1]).reparentTo(base.render)
                    # yhx.p3dh.genpointcloudnodepath(rm.homotransformpointarray(inithomomat, handpalmtemplate), colors=[.5,1,.5,1]).reparentTo(base.render)
                    # yhx.base.run()
                    hto3d = o3dh.nparray_to_o3dpcd(rm.homotransformpointarray(inithomomat, handpalmtemplate))
                    rpo3d = o3dh.nparray_to_o3dpcd(realpcdcrop)
                    inlinnerrmse, newhomomat = o3dh.registration_icp_ptpt(hto3d, rpo3d, np.eye(4),
                                                                          maxcorrdist=2, toggledebug=False)
                    print(inlinnerrmse, ", one round is done!")
                    newhomomatlist.append(rm.homoinverse(newhomomat))
    newhomomat = rm.homomat_average(newhomomatlist, denoise=False)
    refinedamat = np.dot(newhomomat, rawamat)
    pickle.dump(refinedamat, open(os.path.join(yhx.path, "datacalibration", "refinedcalibmat.pkl"), "wb"))
    print(rawamat)
    print(refinedamat)
    return refinedamat


def get_amat(amat_path="calibmat.pkl"):
    filepath = os.path.dirname(os.path.abspath(__file__))+"\\"+amat_path
    amat = pickle.load(open(filepath, "rb"))
    return amat

def transformpcd(amat, pcd):
    """

    :param amat:
    :param pcd:
    :return:

    author: weiwei
    date: 20191228osaka
    """

    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(amat, homopcd).T

    return realpcd[:, :3]

if __name__ == '__main__':
    import os
    import pickle
    import robotconn.rpc.phoxi.phoxi_client as pcdt

    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    armname="rgt"
    yhx = robothelper.RobotHelperX(usereal=True)
    yhx.env.reparentTo(yhx.base.render)

    pxc = pcdt.PhxClient(host="192.168.125.100:18300")
    # relpos = phoxi_computeboardcenterinhand(yhx, pxc, arm_name, parameters, aruco_dict)
    # phoxi_calibbyestinee(yhx, pxc, arm_name, parameters, aruco_dict)
    relpos = np.array([-12.03709376, 1.22814887, 37.36035265])
    phoxi_calib(yhx, pxc, armname, relpos, parameters, aruco_dict)
    # rawamat = pickle.load(open(os.path.join(yhx.path, "datacalibration", "calibmat.pkl"), "rb"))
    # phoxi_calib_refine(yhx, pxc, rawamat, arm_name="rgt")
    # yhx.robot_s.movearmfk(yhx.getarmjntsx("rgt"), arm_name="rgt")
    # yhx.robot_s.movearmfk(yhx.getarmjntsx("lft"), arm_name="lft")
    # rbtnp = yhx.rbtmesh.genmnp(yhx.robot_s)
    # rbtnp.reparentTo(base.render)
    #
    # amat = get_amat()
    # pxc.triggerframe()
    # mph = pxc.getpcd()
    # homopcd = np.ones((4, len(mph)))
    # homopcd[:3, :] = mph.T
    # realpcd = np.dot(amat, homopcd).T
    # pcdnp = yhx.p3dh.genpointcloudnodepath(realpcd)
    # pcdnp.reparentTo(base.render)

    base.run()

    # def update(mph, task):
    #     pcddnp = base.pg.genpointcloudnp(mph)
    #     pcddnp.reparentTo(base.render)
    #     return task.done
