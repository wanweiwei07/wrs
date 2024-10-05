import cv2
import yaml
from cv2 import aruco
import wrs.basis.robot_math as rm


def trackobject_multicamfusion(camcaps, cammtxs, camdists, camrelhomos, aruco_dict, arucomarkersize=100, nframe=5,
                               denoise=True, bandwidth=10):
    """

    :param camcaps: a list of cv2.VideoCaptures
    :param cammtxs: a list of mtx for each of the camcaps
    :param camdists: as list of dist for each of the camcaps
    :param camrelhomos: a list of relative homogeneous matrices
    :param aruco_dict: NOTE this is not things like aruco.DICT_6x6_250, instead, it is the return value of aruco.Dictionary_get
    :param nframe: number of frames used for fusion
    :param denoise:
    :param bandwidth: the bandwidth for meanshift, a large bandwidth leads to tracking instead of clustering, a small bandwith will be very costly
    :return:

    author: weiwei
    date: 20190422
    """

    parameters = aruco.DetectorParameters_create()

    framelist = {}
    for i in range(nframe):
        for capid, cap in enumerate(camcaps):
            ret, frame = cap.read()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters,
                                                                  ids=rm.np.array([[1, 2, 3, 4]]))
            ids = ids.get()
            if ids is not None:
                rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, arucomarkersize, cammtxs[capid],
                                                                           camdists[capid])
                for i in range(ids.size):
                    rot = cv2.Rodrigues(rvecs[i])[0]
                    pos = tvecs[i][0].ravel()
                    if capid > 0:
                        matinb = rm.homomat_inverse(camrelhomos[capid - 1]).dot(rm.homomat_from_posrot(pos, rot))
                        rot = matinb[:3, :3]
                        pos = matinb[:3, 3]
                    idslist = ids.ravel().tolist()
                    if idslist[i] in framelist:
                        framelist[idslist[i]].append([pos, rot])
                    else:
                        framelist[idslist[i]] = [[pos, rot]]

    import time
    frameavglist = {}
    for id in framelist:
        posveclist = [frame[0] for frame in framelist[id]]
        rotmatlist = [frame[1] for frame in framelist[id]]
        if len(posveclist) >= nframe:
            posvecavg = rm.pos_average(posveclist, bandwidth, denoise)
            rotmatavg = rm.rotmat_average(rotmatlist, bandwidth, denoise)
            frameavglist[id] = [posvecavg, rotmatavg]

    return frameavglist


if __name__ == '__main__':
    from wrs import wd

    # square_markersize = 40
    #
    # calibcharucoboard(7,5, square_markersize=square_markersize, imgs_path='./camimgs0/', save_name='cam0_calib.yaml')
    # calibcharucoboard(7,5, square_markersize=square_markersize, imgs_path='./camimgs2/', save_name='cam2_calib.yaml')
    # calibcharucoboard(7,5, square_markersize=square_markersize, imgs_path='./camimgs4/', save_name='cam4_calib.yaml')

    # find_rhomo(base_cam_calibyaml = 'cam0_calib.yaml', rel_cam_calibyaml = 'cam2_calib.yaml', save_name = 'homo_rb20.yaml')
    # find_rhomo(base_cam_calibyaml = 'cam0_calib.yaml', rel_cam_calibyaml = 'cam4_calib.yaml', save_name = 'homo_rb40.yaml')

    base = wd.World(cam_pos=rm.np.array([2.7, 0.3, 2.7]), lookat_pos=rm.np.zeros(3))
    # framenp = base.pggen.genAxis()
    # framenp.reparentTo(base.render)
    # base.run()

    base.pggen.plotAxis(base.render)

    homo_rb20 = yaml.load(open('homo_rb20.yaml', 'r'), Loader=yaml.UnsafeLoader)
    homo_rb40 = yaml.load(open('homo_rb40.yaml', 'r'), Loader=yaml.UnsafeLoader)

    # draw in 3d to validate
    pandamat4homo_r2 = base.pg.np4ToMat4(rm.homoinverse(homo_rb20))
    base.pggen.plotAxis(base.render, spos=pandamat4homo_r2.getRow3(3), pandamat3=pandamat4homo_r2.getUpper3())

    pandamat4homo_r4 = base.pg.np4ToMat4(rm.homoinverse(homo_rb40))
    base.pggen.plotAxis(base.render, spos=pandamat4homo_r4.getRow3(3), pandamat3=pandamat4homo_r4.getUpper3())

    # show in videos
    mtx0, dist0, rvecs0, tvecs0, candfiles0 = yaml.load(open('cam0_calib.yaml', 'r'), Loader=yaml.UnsafeLoader)
    mtx2, dist2, rvecs2, tvecs2, candfiles2 = yaml.load(open('cam2_calib.yaml', 'r'), Loader=yaml.UnsafeLoader)
    mtx4, dist4, rvecs4, tvecs4, candfiles4 = yaml.load(open('cam4_calib.yaml', 'r'), Loader=yaml.UnsafeLoader)

    import time

    # marker_size = int(40*.57)
    # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    cap0 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    cap4 = cv2.VideoCapture(4)

    camcaps = [cap0, cap2, cap4]
    cammtxs = [mtx0, mtx2, mtx4]
    camdists = [dist0, dist2, dist4]
    camrelhomos = [homo_rb20, homo_rb40]
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    arucomarkersize = 100
    nframe = 2
    denoise = True

    framenplist = [[]]


    def updateview(framenplist, task):
        if len(framenplist[0]) > 0:
            for axisnp in framenplist[0]:
                axisnp.removeNode()
        framenplist[0] = []
        tic = time.time()
        frameavglist = trackobject_multicamfusion(camcaps, cammtxs, camdists, camrelhomos, aruco_dict, arucomarkersize,
                                                  nframe, denoise, bandwidth=arucomarkersize * .05)
        print(time.time() - tic)
        for id in frameavglist:
            posvecavg = frameavglist[id][0]
            rotmatavg = frameavglist[id][1]
            framenp = base.pggen.genAxis(spos=base.pg.npToV3(posvecavg), pandamat3=base.pg.npToMat3(rotmatavg))
            framenp.reparentTo(base.render)
            framenplist[0].append(framenp)
        return task.again


    taskMgr.doMethodLater(0.01, updateview, "updateview", extraArgs=[framenplist], appendTask=True)
    base.run()
