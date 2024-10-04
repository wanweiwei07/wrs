import cv2
import yaml
import glob
import numpy as np
from cv2 import aruco
from sklearn import cluster
from wrs import basis as rm, vision as cu


def calibrate_chessboard(ncrossrow,
                         ncrosscolumn,
                         markersize = 25,
                         imgspath='./',
                         savename='mycam_data.yaml'):
    """
    :param ncrossrow: the number of cross in row motion_vec
    :param ncrosscolumn: the number of cross in column motion_vec
    :param markersize: mm
    :param imgspath:
    :param savename:
    :return:
    author: weiwei
    date: 20190420
    """
    worldpoints = cu.gen_world_cross_positions_for_chess(ncrossrow, ncrosscolumn, markersize)
    objpoints = []
    imgpoints = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 27, 0.001)
    images = glob.glob(imgspath+'*.png')
    candfiles=[]
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ncrossrow ,ncrosscolumn), None)
        # If found, add object points, image points (after refining them)
        if ret == True and (len(corners) == ncrossrow*ncrosscolumn):
            corners2 = cv2.cornerSubPix(gray, corners, winSize=(7, 7), zeroZone=(-1, -1), criteria=criteria)
            objpoints.append(worldpoints)
            imgpoints.append(corners2)
            candfiles.append((fname.split("/")[-1]).split("\\")[-1])
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (ncrossrow ,ncrosscolumn), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
    if ret:
        with open(savename, "w") as f:
            yaml.dump([mtx, dist, rvecs, tvecs, candfiles], f)

def calib_charucoboard(nrow,
                       ncolumn,
                       aruco_markerdict=aruco.DICT_6X6_250,
                       square_markersize=25,
                       imgs_path='./',
                       img_format='png',
                       save_name='mycam_charuco_data.yaml'):
    """
    :param nrow:
    :param ncolumn:
    :param marker_dict:
    :param imgs_path:
    :param save_name:
    :return:
    author: weiwei
    date: 20190420
    """
    # read images and detect cornders
    aruco_dict = aruco.Dictionary_get(aruco_markerdict)
    allCorners = []
    allIds = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 27, 0.0001)
    board = aruco.CharucoBoard_create(ncolumn, nrow, square_markersize, .57 * square_markersize, aruco_dict)
    print(imgs_path)
    images = glob.glob(imgs_path + '*.' + img_format)
    candfiles=[]
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize=(2, 2), zeroZone=(-1, -1), criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            # require len(res2[1]) > nrow*ncolumn/2 at least half of the corners are detected
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > (nrow-1)*(ncolumn-1)/2:
                allCorners.append(res2[1])
                allIds.append(res2[2])
                candfiles.append((fname.split("/")[-1]).split("\\")[-1])
            imaxis = aruco.drawDetectedMarkers(img, corners, ids)
            imaxis = aruco.drawDetectedCornersCharuco(imaxis, res2[1], res2[2], (255,255,0))
            cv2.imshow('img', imaxis)
            cv2.waitKey(100)
    # The calibratedCameraCharucoExtended function additionally estimate calibration errors
    # Thus, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors are returned
    # We dont use them here though
    # see https://docs.opencv.org/3.4.6/d9/d6a/group__aruco.html for details
    (ret, mtx, dist, rvecs, tvecs,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors) = \
        cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=allCorners, charucoIds=allIds, board=board,
                                                 imageSize=gray.shape, cameraMatrix=None, distCoeffs=None,
                                                 flags=cv2.CALIB_RATIONAL_MODEL,
                                                 criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print(ret, mtx, dist, rvecs, tvecs, candfiles)
    if ret:
        with open(save_name, "w") as f:
            yaml.dump([mtx, dist, rvecs, tvecs, candfiles], f)

def find_rhomo(base_cam_calibyaml, rel_cam_calibyaml, save_name):
    """
    compute the _rative transformation matrix (homogenous) between two cameras
    the resulting matrix will be:
    resulthomomat*p_in_rel=p_in_base
    :param base_cam_calibyaml: results of board calibration, with (rvecs, tvecs, and candfiles) included
    :param rel_cam_calibyaml: results of board calibration, with (rvecs, tvecs, and candfiles) included
        format of base and rel yaml: [mtx, dist, rvecs, tvecs, candfiles] see calibcharucoboard for example
    :param save_name:
    :return:
    author: weiwei
    date: 20190421
    """
    mtx_b, dist_b, rvecs_b, tvecs_b, candfiles_b = yaml.load(open(base_cam_calibyaml, 'r'), Loader=yaml.UnsafeLoader)
    mtx_r, dist_r, rvecs_r, tvecs_r, candfiles_r = yaml.load(open(rel_cam_calibyaml, 'r'), Loader=yaml.UnsafeLoader)
    # get a list of relative mat from rel to base
    homo_rb_list = []
    for id_b, candimg_b in enumerate(candfiles_b):
        try:
            id_r = candfiles_r.index(candimg_b)
            rot_b, _ = cv2.Rodrigues(rvecs_b[id_b].ravel())
            pos_b = tvecs_b[id_b].ravel()
            homo_b = rm.homobuild(pos_b, rot_b)
            rot_r, _ = cv2.Rodrigues(rvecs_r[id_r].ravel())
            pos_r = tvecs_r[id_r].ravel()
            homo_r = rm.homobuild(pos_r, rot_r)
            # homo_rb*homo_r = homo_b
            homo_rb = np.dot(homo_b, rm.homoinverse(homo_r))
            homo_rb_list.append(homo_rb)
        except Exception as e:
            print(e)
            continue
    # compute the average
    pos_rb_list = []
    quat_rb_list = []
    angle_rb_list = []
    for homo_rb in homo_rb_list:
        quat_rb_list.append(rm.quaternion_from_matrix(homo_rb[:3,:3]))
        angle_rb_list.append([rm.quaternion_angleaxis(quat_rb_list[-1])[0]])
        pos_rb_list.append(homo_rb[:3,3])
    quat_rb_arr = np.array(quat_rb_list)
    mt = cluster.MeanShift()
    quat_rb_arrrefined = quat_rb_arr[np.where(mt.fit(angle_rb_list).labels_==0)]
    quat_rb_avg = rm.quaternion_average(quat_rb_arrrefined)
    pos_rb_avg = mt.fit(pos_rb_list).cluster_centers_[0]
    homo_rb_avg = rm.quaternion_matrix(quat_rb_avg)
    homo_rb_avg[:3,3] = pos_rb_avg
    with open(save_name, "w") as f:
        yaml.dump(rm.homoinverse(homo_rb_avg), f)

if __name__=='__main__':
    # square_markersize = 40
    #
    # calibcharucoboard(7,5, square_markersize=square_markersize, imgs_path='./camimgs0/', save_name='cam0_calib.yaml')
    # calibcharucoboard(7,5, square_markersize=square_markersize, imgs_path='./camimgs2/', save_name='cam2_calib.yaml')
    # calibcharucoboard(7,5, square_markersize=square_markersize, imgs_path='./camimgs4/', save_name='cam4_calib.yaml')

    arucomarkersize = int(40*.57)

    # find_rhomo(base_cam_calibyaml = 'cam0_calib.yaml', rel_cam_calibyaml = 'cam2_calib.yaml', save_name = 'homo_rb20.yaml')
    # find_rhomo(base_cam_calibyaml = 'cam0_calib.yaml', rel_cam_calibyaml = 'cam4_calib.yaml', save_name = 'homo_rb40.yaml')

    import math

    homo_rb20 = yaml.load(open('homo_rb20.yaml', 'r'), Loader=yaml.UnsafeLoader)
    homo_rb40 = yaml.load(open('homo_rb40.yaml', 'r'), Loader=yaml.UnsafeLoader)

    # draw in 3d to validate
    from pandaplotutils import pandactrl
    base = pandactrl.World(camp=[2700, 300, 2700], lookatpos=[0, 0, 0])
    homo_b = rm.homobuild(pos=np.ones(3), rot=np.eye(3))
    base.pggen.plotAxis(base.render)
    pandamat4homo_r2 = base.pg.np4ToMat4(rm.homoinverse(homo_rb20))
    base.pggen.plotAxis(base.render, spos = pandamat4homo_r2.getRow3(3), pandamat3 = pandamat4homo_r2.getUpper3())

    pandamat4homo_r4 = base.pg.np4ToMat4(rm.homoinverse(homo_rb40))
    base.pggen.plotAxis(base.render, spos = pandamat4homo_r4.getRow3(3), pandamat3 = pandamat4homo_r4.getUpper3())
    # base.run()

    # show in videos
    mtx0, dist0, rvecs0, tvecs0, candfiles0 = yaml.load(open('cam0_calib.yaml', 'r'), Loader=yaml.UnsafeLoader)
    mtx2, dist2, rvecs2, tvecs2, candfiles2 = yaml.load(open('cam2_calib.yaml', 'r'), Loader=yaml.UnsafeLoader)
    mtx4, dist4, rvecs4, tvecs4, candfiles4 = yaml.load(open('cam4_calib.yaml', 'r'), Loader=yaml.UnsafeLoader)

    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    cap4 = cv2.VideoCapture(4)

    print(np.linalg.norm(homo_rb20[:3,3]))
    print(np.linalg.norm(homo_rb40[:3,3]))

    while (True):
        ret, frame0 = cap0.read()
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame0, aruco_dict, parameters=parameters)
        if ids is not None:
            aruco.drawDetectedMarkers(frame0, corners, borderColor=[255,255,0])
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, arucomarkersize, mtx0, dist0)
            for i in range(ids.size):
                aruco.drawAxis(frame0, mtx0, dist0, rvecs[i], tvecs[i]/1000.0, 0.1)
                rot = cv2.Rodrigues(rvecs[i])[0]
                pos = tvecs[i][0].ravel()
                base.pggen.plotAxis(base.render, spos = base.pg.npToV3(pos), pandamat3 = base.pg.npToMat3(rot))
        ret, frame2 = cap2.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame2, aruco_dict, parameters=parameters)
        if ids is not None:
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, arucomarkersize, mtx2, dist2)
            for i in range(ids.size):
                aruco.drawAxis(frame2, mtx2, dist2, rvecs[i], tvecs[i]/1000.0, 0.1)
                rot = cv2.Rodrigues(rvecs[i])[0]
                pos = tvecs[i].ravel()
                # base.pggen.plotAxis(base.render, spos = base.pg.npToV3(pos), pandamat3 = base.pg.npToMat3(rotmat), rgba=[1,0,1,1])
                matinb = np.dot(rm.homoinverse(homo_rb20), rm.homobuild(pos, rot))
                rot = matinb[:3,:3]
                pos = matinb[:3,3]
                base.pggen.plotAxis(base.render, spos = base.pg.npToV3(pos), pandamat3 = base.pg.npToMat3(rot), rgba=[1,1,0,1])
                angle, axis = rm.quaternion_angleaxis(rm.quaternion_from_matrix(rot))
                rvecnew = np.array([math.radians(angle)*axis])
                tvecnew = np.array([pos])
                rot = cv2.Rodrigues(rvecnew)[0]
                pos = tvecnew.ravel()
                aruco.drawAxis(frame0, mtx2, dist2, rvecnew, tvecnew/1000.0, 0.1)
        ret, frame4 = cap4.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame4, aruco_dict, parameters=parameters)
        if ids is not None:
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, arucomarkersize, mtx4, dist4)
            for i in range(ids.size):
                aruco.drawAxis(frame4, mtx4, dist4, rvecs[i], tvecs[i]/1000.0, 0.1)
                rot = cv2.Rodrigues(rvecs[i])[0]
                pos = tvecs[i].ravel()
                # base.pggen.plotAxis(base.render, spos = base.pg.npToV3(pos), pandamat3 = base.pg.npToMat3(rotmat), rgba=[1,0,1,1])
                matinb = np.dot(rm.homoinverse(homo_rb40), rm.homobuild(pos, rot))
                rot = matinb[:3,:3]
                pos = matinb[:3,3]
                base.pggen.plotAxis(base.render, spos = base.pg.npToV3(pos), pandamat3 = base.pg.npToMat3(rot), rgba=[1,1,0,1])
                angle, axis = rm.quaternion_angleaxis(rm.quaternion_from_matrix(rot))
                rvecnew = np.array([math.radians(angle)*axis])
                tvecnew = np.array([pos])
                rot = cv2.Rodrigues(rvecnew)[0]
                pos = tvecnew.ravel()
                aruco.drawAxis(frame0, mtx4, dist4, rvecnew, tvecnew/1000.0, 0.1)

        base.run()
        cv2.imshow('frame0', frame0)
        cv2.moveWindow('rgt', 200, 800)
        cv2.imshow('frame2', frame2)
        cv2.moveWindow('rgt', 900, 800)
        cv2.imshow('frame4', frame4)
        cv2.moveWindow('rgt', 1600, 800)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    #
    # rvecs0dataset=[]
    # tvecs0dataset=[]
    # rvecs2dataset=[]
    # tvecs2dataset=[]
    # rvecs4dataset=[]
    # tvecs4dataset=[]
    # homo20list = []
    # homo40list = []
    # for id0, candimg0 in enumerate(candfiles0):
    #     try:
    #         id2 = candfiles2.index(candimg0)
    #         id4 = candfiles4.index(candimg0)
    #
    #         rvecs0dataset.append(rvecs0[id0].ravel())
    #         tvecs0dataset.append(tvecs0[id0].ravel())
    #         rot0, _ = cv2.Rodrigues(rvecs0dataset[-1])
    #         pos0 = tvecs0dataset[-1]
    #         homo0 = rm.homobuild(pos0, rot0)
    #
    #         rvecs2dataset.append(rvecs2[id2].ravel())
    #         tvecs2dataset.append(tvecs2[id2].ravel())
    #         rot2, _ = cv2.Rodrigues(rvecs2dataset[-1])
    #         pos2 = tvecs2dataset[-1]
    #         homo2 = rm.homobuild(pos2, rot2)
    #
    #         rvecs4dataset.append(rvecs4[id4].ravel())
    #         tvecs4dataset.append(tvecs4[id4].ravel())
    #         rot4, _ = cv2.Rodrigues(rvecs4dataset[-1])
    #         pos4 = tvecs4dataset[-1]
    #         homo4 = rm.homobuild(pos4, rot4)
    #
    #         # homo20*homo2 = homo0
    #         homo20 = np.dot(homo0, rm.homoinverse(homo2))
    #         homo40 = np.dot(homo0, rm.homoinverse(homo4))
    #         homo20list.append(homo20)
    #         homo40list.append(homo40)
    #     except Exception as e:
    #         print(e)
    #         continue
    #
    # from pandaplotutils import pandactrl
    # base = pandactrl.World(camp=[2700, 300, 2700], lookatp=[0, 0, 0])
    # homo0 = rm.homobuild(pos=np.ones(3), rotmat=np.eye(3))
    # base.pggen.plotAxis(base.render)
    # for homo20 in homo20list:
    #     homo2 = np.dot(rm.homoinverse(homo20), homo0)
    #     pandamat4homo2 = base.pg.np4ToMat4(homo2)
    #     base.pggen.plotAxis(base.render, spos = pandamat4homo2.getRow3(3), pandamat3 = pandamat4homo2.getUpper3(), rgba = [.3,.5,.5,.1])
    # for homo40 in homo40list:
    #     homo4 = np.dot(rm.homoinverse(homo40), homo0)
    #     pandamat4homo4 = base.pg.np4ToMat4(homo4)
    #     base.pggen.plotAxis(base.render, spos = pandamat4homo4.getRow3(3), pandamat3 = pandamat4homo4.getUpper3(), rgba = [.5,.5,.3,.1])
    #
    # pos2list = []
    # quat2list = []
    # angle2list = []
    # for homo20 in homo20list:
    #     homo2 = np.dot(rm.homoinverse(homo20), homo0)
    #     quat2list.append(rm.quaternion_from_matrix(homo2[:3,:3]))
    #     angle2list.append([rm.quaternion_angleaxis(quat2list[-1])[0]])
    #     pos2list.append(homo2[:3,3])
    # quat2arr = np.array(quat2list)
    # mt = cluster.MeanShift()
    # quat2arrrefined = quat2arr[np.where(mt.fit(angle2list).labels_==0)]
    # quat2avg = rm.quaternion_average(quat2arrrefined)
    # pos2avg = mt.fit(pos2list).cluster_centers_[0]
    # homo2avg = rm.quaternion_matrix(quat2avg)
    # homo2avg[:3,3] = pos2avg
    # pandamat4homo21 = base.pg.np4ToMat4(homo2avg)
    #
    # # import matplotlib.pyplot as plt
    # # plt.plot(pos2list)
    # # plt.plot(pos2avg)
    # # plt.show()
    #
    # pos4list = []
    # quat4list = []
    # angle4list = []
    # for homo40 in homo40list:
    #     homo4 = np.dot(rm.homoinverse(homo40), homo0)
    #     quat4list.append(rm.quaternion_from_matrix(homo4[:3,:3]))
    #     angle4list.append([rm.quaternion_angleaxis(quat4list[-1])[0]])
    #     pos4list.append(homo4[:3,3])
    # quat4arr = np.array(quat4list)
    # mt = cluster.MeanShift()
    # quat4arrrefined = quat4arr[np.where(mt.fit(angle4list).labels_==0)]
    # quat4avg = rm.quaternion_average(quat4arrrefined)
    # pos4avg = mt.fit(pos4list).cluster_centers_[0]
    # homo4avg = rm.quaternion_matrix(quat4avg)
    # homo4avg[:3,3] = pos4avg
    # pandamat4homo4 = base.pg.np4ToMat4(homo4avg)
    #
    # base.pggen.plotAxis(base.render, spos = pandamat4homo2.getRow3(3), pandamat3 = pandamat4homo2.getUpper3())
    # base.pggen.plotAxis(base.render, spos = pandamat4homo4.getRow3(3), pandamat3 = pandamat4homo4.getUpper3())
    #
    # # p0 = homo20*p2
    # # homo20 = inv(homo2)
    # savename20 = "homo20.yaml"
    # with open(savename20, "w") as f:
    #     yaml.dump(rm.homoinverse(homo2avg), f)
    # savename40 = "homo40.yaml"
    # with open(savename40, "w") as f:
    #     yaml.dump(rm.homoinverse(homo4avg), f)
    #
    #
    # base.run()