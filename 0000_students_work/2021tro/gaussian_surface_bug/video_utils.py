import os

import cv2
import cv2.aruco as aruco
import numpy as np

import config

cameraMatrix = np.array([[1.42068235e+03, 0.00000000e+00, 9.49208512e+02],
                         [0.00000000e+00, 1.37416685e+03, 5.39622051e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([1.69926613e-01, -7.40003491e-01, -7.45655262e-03, -1.79442353e-03, 2.46650225e+00])


def video2img(video_f_name):
    vc = cv2.VideoCapture(os.path.join(config.ROOT, "video", video_f_name))
    c = 1

    output_path = os.path.join(config.ROOT, "img/videocapture/")
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        print("open error!")
        rval = False

    time_interval = 5
    while rval:
        rval, frame = vc.read()
        if c % time_interval == 0:
            if frame is None:
                continue
            corners, ids = detect_aruco(frame)
            if corners is None:
                continue
            print(ids)
            for i, corner in enumerate(corners):
                points = corner[0].astype(np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 255))
                cv2.putText(frame, str(ids[i][0]), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.3, cameraMatrix, distCoeffs)

            for i in range(ids.size):
                # print( 'rvec {}, tvec {}'.format( rvecs[i], tvecs[i] ))
                # print( 'rvecs[{}] {}'.format( i, rvecs[i] ))
                # print( 'tvecs[{}] {}'.format( i, tvecs[i] ))
                aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1)
                cv2.imshow("img", frame)
            cv2.waitKey(0)
            # cv2.imwrite(output_path+video_f_name.split(".mp4")[0] + str(int(c / time_intervals)) + '.jpg', frame)
        c += 1
    vc.release()


def detect_aruco(img, tgtids=[1, 3, 9, 619]):
    parameters = aruco.DetectorParameters_create()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    corners, ids, _ = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.3, cameraMatrix, distCoeffs)

    if len(corners) < len(tgtids):
        return None, None
    if len(ids) != len(tgtids):
        return None, None
    if ids[0] not in tgtids or ids[1] not in tgtids:
        return None, None
    return corners, ids


if __name__ == '__main__':
    video_f_name = 'cat.mp4'
    video2img(video_f_name)
