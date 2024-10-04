import wrs.vision.ar_marker.make_pdfboard as mpb
import cv2.aruco as aruco

mpb.make_aruco_board(2,1,marker_dict=aruco.DICT_4X4_250, start_id=2, marker_size=120)